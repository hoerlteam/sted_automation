import logging
from itertools import chain
import heapq
from collections import defaultdict
from time import time, sleep
import os
import hashlib

from autosted.imspector import ImspectorConnection
from autosted.data import MeasurementData, HDF5DataStore
from autosted.utils.delayed_interrupt import DelayedKeyboardInterrupt
from autosted.stoppingcriteria.stoppingcriteria import InterruptedStoppingCriterion
from autosted.taskgeneration.taskgeneration import AcquisitionTask


class AcquisitionPipeline:

    # keep reference to currently running instance
    running_instance = None

    def __init__(
        self,
        data_save_path,
        hierarchy_levels,
        imspector=None,
        save_combined_hdf5=False,
        level_priorities=None,
        file_prefix=None,
    ):
        """
        An `AcquisitionPipeline` is the main object of an autoSTED acquisition pipeline run.

        Parameters
        ----------
        data_save_path (str):
            The path where the acquired data will be saved.
        hierarchy_levels (list of str):
            names of the levels in a hierarchical acquisition (e.g. ["overview", "detail"])
            Acquisition Tasks with a later level in this list will have higher priority
        imspector (Imspector, optional):
            An instance of SpcPy `Imspector` class to use to interface with the microscope.
            By default, we make a new instance connecting to a local Imspector.
        save_combined_hdf5 (bool, optional):
            A flag indicating whether to save the all acquired data in one HDF5 file
            (in addition to the individual .msr files). Default is False.
        level_priorities (dict of str->int, optional):
            Assign different priorities to levels of the acquisition (lower values will be prioritized).
        file_prefix (str, optional):
            A prefix for the filenames of the saved data files.
            Default is None (a random alphanumeric string will be used).
        """

        # name of the run, mostly unused at the moment
        self.name = "automatic-acquisition"

        self.hierarchy_levels = hierarchy_levels

        # by default, priorities are the inverse of the order of hierarchy_levels
        # e.g., details have higher priority than overviews
        if level_priorities is None:
            level_priorities = dict(
                zip(reversed(self.hierarchy_levels), range(len(self.hierarchy_levels)))
            )
        self.level_priorities = level_priorities

        # we have an InterruptedStoppingCriterion by default
        self.stopping_conditions = [InterruptedStoppingCriterion()]

        # the queue is just a list, but should only be modified via the heapq module
        self.queue = []

        # levels at which new tasks will reuse the index of their parent level instead of starting from 0
        # use e.g. to give coarse overview -> fine overview of same FOV the same id
        self._levels_reusing_parent_index = []

        # levels that should not show up in filnames for files at other levels
        # e.g. acquisitions just for autofocus (levels: autofocus, image, detail)
        # filenames will be imageX_detailY instead of autofocusX_imageY_detailZ
        # TODO: mask levels in H5 as well
        self._masked_levels_in_filename = []

        # keep track of starting time, e.g. for TimedStoppingCriterion
        self.starting_time = None

        # hold the Imspector connection
        if imspector is not None:
            self.imspector_connection = ImspectorConnection(imspector)
        # delay if Imspector is not explicitly given
        else:
            self.imspector_connection = None

        self.logger = logging.getLogger(__name__)

        # set up file name handling and create output directory
        self.base_path = os.path.abspath(data_save_path)
        self.filename_handler = FilenameHandler(
            self.base_path, file_prefix
        )
        # make directory if it does not exist yet
        if not os.path.exists(self.base_path):
            os.makedirs(self.base_path)

        # set up data storage, can be hdf5 or just an in-memory defaultdict
        if save_combined_hdf5:
            self.data = HDF5DataStore(
                self.filename_handler.get_path((), ".h5")
            )
        else:
            self.data = defaultdict(MeasurementData)

        # callbacks are stored in a dict level_name -> list of callbacks
        self.callbacks = defaultdict(list)

        # keep track of last time measurements of each level were started
        self.last_measurement_start_times = defaultdict(float)

        # boolean flag member, the DelayedKeyboardInterrupt will indicate a received SIGINT here
        self.interrupted = False

    def run(self, initial_callback=None):
        """
        run the pipeline
        """

        # this instance is now the currently running one
        if AcquisitionPipeline.running_instance is not None:
            self.logger.warning(
                "Another pipeline instance is currently running, this is likely to cause conflicts."
            )
        AcquisitionPipeline.running_instance = self

        # init Imspector connection if we didn't already
        # e.g. it could have been replaced by a simulated demo connection
        if self.imspector_connection is None:
            self.imspector_connection = ImspectorConnection(None)

        # we use this context manager to handle interrupts,
        # so we can finish the acquisition we are in before stopping
        with DelayedKeyboardInterrupt(self):

            # record starting time, so we can check whether a StoppingCondition is met
            self.starting_time = time()

            # run initial callback to populate queue
            if initial_callback is not None:
                new_level, new_tasks = initial_callback()
                for task in new_tasks:
                    # add with empty parent idx
                    self.enqueue_task(new_level, task, ())

            # main acquisition loop
            while len(self.queue) > 0:

                # pop next task from queue
                priority, index, acquisition_task = heapq.heappop(self.queue)

                # get level of current task (last (level, idx) of index)
                current_level, _ = index[-1]

                # if we have an actual AcquisitionTask wrapper object, get it's delay
                # otherwise (e.g. we have list of parameters) default to 0
                delay = (
                    acquisition_task.delay
                    if isinstance(acquisition_task, AcquisitionTask)
                    else 0
                )

                if delay > 0:
                    wait_time = max(
                        0,
                        delay
                        - (time() - self.last_measurement_start_times[current_level]),
                    )
                    self.logger.info(
                        f"Waiting {wait_time:.3f} seconds until next acquisition at level {current_level} (for delay of {delay})."
                    )
                    sleep(wait_time)

                self.last_measurement_start_times[current_level] = time()

                # even if we have an empty dummy acquisition task, add empty data to storage
                # otherwise, indices may be re-used and lead to weird phenomena
                if len(acquisition_task) == 0:
                    self.data[index]

                # go through updates sequentially (we might have multiple configurations per measurement)
                for update_index in range(len(acquisition_task)):

                    # make measurement (for first update) or configuration (for subsequent) in Imspector
                    if update_index == 0:
                        self.imspector_connection.make_measurement_from_task(
                            acquisition_task[update_index]
                        )
                    else:
                        self.imspector_connection.make_configuration_from_task(
                            acquisition_task[update_index]
                        )

                    # run in Imspector
                    self.imspector_connection.run_current_measurement()

                    # add data copy (of most recent configuration) to data storage
                    self.data[index].append(
                        *self.imspector_connection.get_current_data()
                    )

                # save and close in Imspector
                # only save if we actually did any acquisitions
                if len(acquisition_task) > 0:

                    # get levels to mask (if current level is in there, remove it)
                    levels_to_mask = self._masked_levels_in_filename
                    if current_level in self._masked_levels_in_filename:
                        levels_to_mask = set(self._masked_levels_in_filename)
                        levels_to_mask.remove(current_level)

                    path = self.filename_handler.get_path(
                        index, mask_levels=levels_to_mask
                    )
                    self.imspector_connection.save_current_measurement(path)
                    self.imspector_connection.close_current_measurement()

                # do the callbacks (this should do analysis and return tasks to re-fill the queue)
                callbacks_for_current_level = self.callbacks.get(current_level, None)
                if not (callbacks_for_current_level is None):
                    for callback in callbacks_for_current_level:
                        # allow callback to return None for flexibility
                        # default (AcquisitionTaskGenerator) will return list of new tasks and their level
                        result = callback()
                        if result is not None:
                            new_level, new_tasks = result
                            for task in new_tasks:
                                self.enqueue_task(
                                    new_level,
                                    task,
                                    index,
                                    new_level in self._levels_reusing_parent_index,
                                )

                # go through stopping conditions
                stopping_condition_met = False
                for sc in self.stopping_conditions:
                    if sc.check(self):
                        print(sc.desc(self))
                        stopping_condition_met = True
                        break

                # break from main loop
                if stopping_condition_met:
                    break

            self.logger.info("PIPELINE {} FINISHED".format(self.name))

            # unset currently running instance
            self.__class__.running_instance = None

    def get_next_free_index(self, hierarchy_level, parent_index=()):
        """
        get the next free index for a task to be added to the queue
        checks already imaged idxs from data and idxs currently in queue to prevent re-use of the same index
        """

        # we are at the same level as parent (e.g. next overview is enqueued after previous)
        # therefore, we ignore the last element of parent
        if len(parent_index) > 0 and parent_index[-1][0] == hierarchy_level:
            parent_index = parent_index[:-1]

        indices = self.get_all_used_indices(hierarchy_level)

        # get all that start with parent index
        # NOTE: only parts of the parent index up to hierarchy level are considered
        # e.g., when an overview is inserted after a callback by the previous overview
        # it should still get a new index
        indices = [
            idx
            for idx in indices
            if idx[:-1] == parent_index
        ]

        # get the indices (acq. numbers) of the last hierarchy level
        index_at_level = [idx[-1][1] for idx in indices]

        # next index is either the maximum found + 1 or 0 if nothing yet at this level
        next_index = max(index_at_level) + 1 if len(index_at_level) > 0 else 0

        # return as tuple of (level, index) pairs
        return parent_index + ((hierarchy_level, next_index),)

    def get_all_used_indices(self, hierarchy_level):
        """
        get a set of all indices with same hierarchy level from data (already imaged and thus taken)
        and from queue (not yet imaged but already enqueued, so also taken)
        """

        # get indices of same hierarchy level from queue and data store
        # indices are (level, number) pairs
        # the last index indicates the level of actual measurement, the others parent indices
        indices_in_queue = {
            idx for prio, idx, task in self.queue if idx[-1][0] == hierarchy_level
        }
        indices_in_data = {
            idx for idx in self.data.keys() if idx[-1][0] == hierarchy_level
        }
        indices = indices_in_queue.union(indices_in_data)
        return indices

    def enqueue_task(self, level, task, parent_index=(), reuse_parent_index=False):

        new_priority = self.level_priorities[level]

        if reuse_parent_index:
            # repeat last value from parent index e.g. (('a', 4), ('b', 2)) -> (('a', 4), ('b', 2), ('c', 2))
            new_index = parent_index + ((level, parent_index[-1][1]),)
            # check if it already exists, warn of overwrite in that case
            if new_index in self.get_all_used_indices(level):
                self.logger.warning(
                    "Reusing index {new_index}, data will be overwritten!"
                )
        else:
            new_index = self.get_next_free_index(level, parent_index)

        heapq.heappush(self.queue, (new_priority, new_index, task))

    def add_callback(self, callback, level):
        """
        add a callback for a hierarchy level
        """
        if level not in self.hierarchy_levels:
            raise ValueError("{} is not a registered pipeline level".format(level))
        self.callbacks[level].append(callback)

    def add_stopping_condition(self, cond):
        """
        add a StoppingCondition
        """
        self.stopping_conditions.append(cond)


class FilenameHandler:
    """
    helper class to generate systematic filenames to save data to.
    """

    random_prefix_length = 8

    def __init__(
        self,
        path,
        prefix=None,
        default_ending=".msr",
        min_index_padding_length=0,
    ):
        self.path = path
        # TODO: remove unnecessary levels parameter
        # self.levels = levels
        self.default_ending = default_ending
        self.min_index_padding_length = min_index_padding_length

        # if no prefix for filenames is given, use a random hash
        if prefix is None:
            hash_object = hashlib.md5(bytes(str(time()), "utf-8"))
            hex_dig = hash_object.hexdigest()
            self.prefix = str(hex_dig)[: self.random_prefix_length]
        else:
            self.prefix = prefix

        # format string used for each (level, index)-pair in filename generation
        self.insert_fstring = "_{}_{}"

    @staticmethod
    def leftpad(string, length, padding_char="0"):
        return padding_char * (length - len(string)) + string

    def get_filename(self, idxes=(), ending=None, mask_levels=None):

        # left-pad to desired length
        idxes = [(level, self.leftpad(str(idx), self.min_index_padding_length)) for level, idx in idxes]

        # get level, index-pairs, drop masked levels if necessary
        if mask_levels is not None:
            insert_pairs = [
                (level, index)
                for level, index in idxes
                if level not in mask_levels
            ]
        else:
            insert_pairs = idxes

        # make chained inserts [level1, idx1, level2, idx2, ...]
        insert = chain.from_iterable(insert_pairs)
        insert = list(insert)

        # if no ending was specified, use default one
        if ending is None:
            ending = ending or self.default_ending

        return (self.prefix + self.insert_fstring * len(insert_pairs)).format(
            *insert
        ) + ending

    def get_path(self, idxes=(), ending=None, mask_levels=None):
        return os.path.join(self.path, self.get_filename(idxes, ending, mask_levels))
