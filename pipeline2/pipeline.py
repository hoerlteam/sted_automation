import logging
from itertools import chain
import heapq
from collections import defaultdict
from time import time
import os
import hashlib

from pipeline2.imspector import ImspectorConnection
from pipeline2.data import MeasurementData, HDF5DataStore
from pipeline2.utils.delayed_interrupt import DelayedKeyboardInterrupt
from pipeline2.stoppingcriteria.stoppingcriteria import InterruptedStoppingCriterion


class AcquisitionPipeline:
    """
    the main class of an acquisition pipeline run
    """
    def __init__(self, name, path, hierarchy_levels, imspector=None, save_combined_hdf5=True, level_priorities=None):

        self.name = name
        self.hierarchy_levels = hierarchy_levels

        # by default, priorities are the inverse of the order of hierarchy_levels
        # e.g., details have higher priority than overviews
        if level_priorities is None:
            level_priorities = dict(zip(reversed(self.hierarchy_levels), range(len(self.hierarchy_levels))))
        self.level_priorities = level_priorities

        # we have an InterruptedStoppingCriterion by default
        self.stopping_conditions = [InterruptedStoppingCriterion()]

        # the queue is just a list, but should only be modified via the heapq module
        self.queue = []

        # keep track of starting time, so
        self.starting_time = None

        # hold the Imspector connection
        self.imspector_connection = ImspectorConnection(imspector)

        self.logger = logging.getLogger(__name__)

        # set up file name handling and create output directory
        self.base_path = path
        self.filename_handler = FilenameHandler(self.base_path, self.hierarchy_levels)
        # make directory if it does not exist yet
        if not os.path.exists(self.base_path):
            os.makedirs(self.base_path)

        # set up data storage, can be hdf5 or just an in-memory defaultdict
        if save_combined_hdf5:
            self.data = HDF5DataStore(self.filename_handler.get_path((), '.h5'), self.hierarchy_levels)
        else:
            self.data = defaultdict(MeasurementData)

        # callbacks are stored in a dict level_name -> list of callbacks
        self.callbacks = defaultdict(list)

        # boolean flag member, the DelayedKeyboardInterrupt will indicate a received SIGINT here
        self.interrupted = False

    def run(self, initial_callback=None):
        """
        run the pipeline
        """

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

                # go through updates sequentially (we might have multiple configurations per measurement)
                for update_index in range(acquisition_task.num_acquisitions):

                    # make measurement (for first update) or configuration (for subsequent) in Imspector
                    if update_index == 0:
                        self.imspector_connection.make_measurement_from_task(acquisition_task.get_updates(update_index, True), acquisition_task.delay)
                    else:
                        self.imspector_connection.make_configuration_from_task(acquisition_task.get_updates(update_index, True), acquisition_task.delay)

                    # run in Imspector
                    self.imspector_connection.run_current_measurement()

                    # add data copy (of most recent configuration) to data storage
                    self.data[index].append(*self.imspector_connection.get_current_data())

                # save and close in Imspector
                # only save if we actually did any acquisitions
                if acquisition_task.num_acquisitions > 0:
                    path = self.filename_handler.get_path(index)
                    self.imspector_connection.save_current_measurement(path)
                    self.imspector_connection.close_current_measurement()

                # get level of current task
                current_level = next(hierarchy_level for hierarchy_level, priority_i in self.level_priorities.items() if priority_i == priority)

                # do the callbacks (this should do analysis and return tasks to re-fill the queue)
                callbacks_for_current_level = self.callbacks.get(current_level, None)
                if not (callbacks_for_current_level is None):
                    for callback in callbacks_for_current_level:
                        new_level, new_tasks = callback(self)
                        for task in new_tasks:
                            self.enqueue_task(new_level, task, index)

                # go through stopping conditions
                stopping_condition_met = False
                for sc in self.stopping_conditions:
                    if sc.check(self):
                        # reset interrupt flag if necessary
                        if isinstance(sc, InterruptedStoppingCriterion):
                            sc.resetInterrupt(self)
                        print(sc.desc(self))
                        stopping_condition_met = True
                        break

                # break from main loop
                if stopping_condition_met:
                    break

            self.logger.info('PIPELINE {} FINISHED'.format(self.name))

    def get_next_free_index(self, hierarchy_level, parent_index=()):
        """
        get the next free index for a task to be added to the queue
        checks already imaged idxs from data and idxs currently in queue to prevent re-use of the same index
        """

        # quick sanity check to ensure we have a long enough parent index to generate a new one
        if len(parent_index) < hierarchy_level:
            raise ValueError("length of parent index must be at least equal to hierarchy level")

        # get all indices with same hierarchy level from data (already imaged and thus taken)
        # and from queue (not yet imaged but already enqueued, so also taken)
        indices_in_queue = {idx for prio, idx, task in self.queue if len(idx) == hierarchy_level + 1}
        indices_in_data = {idx for idx in self.data.keys() if len(idx) == hierarchy_level + 1}
        indices = indices_in_queue.union(indices_in_data)

        # get all that start with parent index
        # NOTE: only parts of the parent index up to hierarchy level are considered
        # e.g., when an overview is inserted after a callback by the previous overview
        # it should still get a new index
        indices = [idx for idx in indices if idx[:hierarchy_level] == parent_index[:hierarchy_level]]

        # get the indices of the last hierarchy level
        index_at_level = [idx[hierarchy_level] for idx in indices]

        # next index is either the maximum found + 1 or 0 if nothing yet at this level
        next_index = max(index_at_level) + 1 if len(index_at_level) > 0 else 0

        # return as index tuple
        return parent_index[:hierarchy_level] + (next_index,)

    def enqueue_task(self, level, task, parent_index):
        new_priority = next(priority for hierarchy_level, priority in self.level_priorities.items() if hierarchy_level == level)
        new_hierarchy_index = self.hierarchy_levels.index(level)
        heapq.heappush(self.queue, (new_priority, self.get_next_free_index(new_hierarchy_index, parent_index), task))

    def add_callback(self, callback, level):
        """
        add a callback for a hierarchy level
        """
        if level not in self.hierarchy_levels:
            raise ValueError('{} is not a registered pipeline level'.format(level))
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
    # TODO: add zero-padding of indices in filenames?

    random_prefix_length = 8

    def __init__(self, path, levels, prefix=None, default_ending ='.msr'):
        self.path = path
        self.levels = levels
        self.default_ending = default_ending

        # if no prefix for filenames is given, use a random hash
        if prefix is None:
            hash_object = hashlib.md5(bytes(str(time()), "utf-8"))
            hex_dig = hash_object.hexdigest()
            self.prefix = str(hex_dig)[:self.random_prefix_length]
        else:
            self.prefix = prefix

        # format string used for each (level, index)-pair in filename generation
        self.insert_fstring = '_{}_{}'

    def get_filename(self, idxes=(), ending=None):

        # make chained inserts [level1, idx1, level2, idx2, ...]
        insert = chain.from_iterable(zip(self.levels[0:len(idxes)], idxes))
        insert = list(insert)

        # if no ending was specified, use default one
        if ending is None:
            ending = ending or self.default_ending

        return (self.prefix + self.insert_fstring * len(idxes)).format(*insert) + ending
    
    def get_path(self, idxes=(), ending=None):
        return os.path.join(self.path, self.get_filename(idxes, ending))


if __name__ == '__main__':
    file_handler = FilenameHandler('/path/to/file', ['overview', 'detail'])
    print(file_handler.get_path((2,3), ending='.h5'))