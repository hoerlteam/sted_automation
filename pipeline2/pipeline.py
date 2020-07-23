from itertools import count, chain

from queue import PriorityQueue
from collections import defaultdict
from time import time, sleep, clock
import os
import hashlib

from .imspector.imspector import MockImspectorConnection
from .data import RichData
from .util import DelayedKeyboardInterrupt
from .stoppingcriteria.stoppingcriteria import InterruptedStoppingCriterion

#from jsonpath_ng import jsonpath, parse

from spot_util import pair_finder_inner

class AcquisitionPriorityQueue(PriorityQueue):
    """
    slightly modified PriorityQueue to be able to enqueue non-orderable data
    """

    def __init__(self):
        PriorityQueue.__init__(self)
        self.ctr = count()

    def put(self, item, prio):
        PriorityQueue.put(self, (prio, next(self.ctr), item))

    def get(self, *args, **kwargs):
        lvl, _, item = PriorityQueue.get(self, *args, **kwargs)
        return (lvl, item)


class _pipeline_level:
    """
    named level in an acquisition pipeline
    should not be used outside of a PipelineLevels object
    """

    def __init__(self, parent, name):
        self.parent = parent
        self.name = name

    def __eq__(self, other):
        if type(self) != type(other):
            return False
        return self.name == other.name

    def __le__(self, other):
        return self.parent.reversedLevels.index(self) <= self.parent.reversedLevels.index(other)

    def __lt__(self, other):
        return self.parent.reversedLevels.index(self) < self.parent.reversedLevels.index(other)

    def __str__(self):
        return self.name

    def __hash__(self):
        return str.__hash__(self.name)

    def __repr__(self):
        return self.name

class PipelineLevels:
    """
    ordered collection of _pipeline_level
    """
    levels = []
    def __init__(self, *args):
        for arg in args:
            lvl = _pipeline_level(self, arg)
            self.levels.append(lvl)
            setattr(self, arg, lvl)
    @property
    def reversedLevels(self):
        return list(reversed(self.levels))

class AcquisitionPipeline():
    """
    the main class
    """

    def __init__(self, name):
        """
        construct with name
        """
        self.name = name

        self.pipelineLevels = None

        # we habe an InterruptedStoppingCriterion by default
        self.stoppingConditions = [InterruptedStoppingCriterion()]
        self.queue = AcquisitionPriorityQueue()
        self.startingTime = None
        self.counters = defaultdict(int)
        self.data = defaultdict(RichData)
        self.callbacks = defaultdict(list)

        # hold the Imspector connection
        self.im = MockImspectorConnection()

        self.logger = None
        self.nameHandler = None

        # the DelayedKeyboardInterrupt will indicate a received SIGINT here
        self.interrupted = False

    def run(self):
        """
        run the pipeline
        """

        # we use this context manager to handle interrupts so we can finish
        # to acquisition we are in before stopping
        with DelayedKeyboardInterrupt(self):

            # record starting time, so we can check wether a StoppingCondition is met
            self.startingTime = time()

            lvl = None

            while not self.queue.empty():

                # get next task and its level
                oldlvl = lvl
                lvl, acquisition_task = self.queue.get()

                if oldlvl is None:
                    self.counters[lvl] = -1

                # reset or increment indices
                if (oldlvl != lvl):
                    for l in self.pipelineLevels.levels:
                        if l < lvl:
                            self.counters[l] = -1

                self.counters[lvl] += 1

                # create index of measurement (indices of all levels until lvl)
                currentMeasurementIdx = tuple([self.counters[l] for l in self.pipelineLevels.levels[
                                                                         0:self.pipelineLevels.levels.index(lvl) + 1]])

                # go through updates sequentially (we might have multiple configurations per measurement)
                for updatesI in range(acquisition_task.numAcquisitions):

                    # update imspector
                    if updatesI == 0:
                        self.im.makeMeasurementFromTask(acquisition_task.getUpdates(updatesI), acquisition_task.delay)
                    else:
                        self.im.makeConfigurationFromTask(acquisition_task.getUpdates(updatesI), acquisition_task.delay)

                    meas_startime = time()

                    # run in imspector
                    self.im.runCurrentMeasurement(acquisition_task.getUpdates(updatesI))

                    meas_endtime = time()


                    # add data copy (of most recent configuration) to internal storage
                    self.data[currentMeasurementIdx].append(*self.im.getCurrentData())

                # save and close in imspector
                path = None
                if self.nameHandler != None:
                    path = self.nameHandler.get_path(currentMeasurementIdx)
                print(path)

                # TODO: closing without saving might trigger UI dialog in Imspector
                if not (path is None):
                    self.im.saveCurrentMeasurement(path)
                self.im.closeCurrentMeasurement()
                
                # NB: give Imspector time to close measurement
                #sleep(3.0)

                # do the callbacks (this should do analysis and re-fill the queue)
                callbacks_ = self.callbacks.get(lvl, None)
                if not (callbacks_ is None):
                    for callback_ in callbacks_:
                        callback_(self)

                # go through stopping conditions
                for sc in self.stoppingConditions:
                    if sc.check(self) == True:
                        # reset interrupt flag if necessary
                        if isinstance(sc, InterruptedStoppingCriterion):
                            sc.resetInterrupt(self)
                        print(sc.desc(self))
                        break
                # we went through all the loop iterations (no break)
                else:
                    continue
                break

            print('PIPELINE {} FINISHED'.format(self.name))

    def withDataStorage(self, data):
        """
        set a custom Data storage
        :param data: data storage object, must act like defaultdict(RichData)
        :return: self
        """
        self.data = data
        return self

    def withPipelineLevels(self, lvls):
        """
        set pipeline levels, can be chained
        """
        self.pipelineLevels = lvls
        return self

    def withNameHandler(self, nh):
        self.nameHandler = nh
        return self

    def withImspectorConnection(self, im):
        self.im = im
        return self

    def withCallbackAtLevel(self, callback, lvl):
        """
        set the callback for a level, can be chained
        """
        if not (lvl in self.pipelineLevels.levels):
            raise ValueError('{} is not a registered pipeline level'.format(lvl))
        self.callbacks[lvl].append(callback)
        return self

    def _withStoppingConditions(self, conds):
        """
        reset the StoppingConditions, can be chained
        """
        self.stoppingConditions.clear()
        for condI in conds:
            self.stoppingConditions.append(condI)
        return self

    def withAddedStoppingCondition(self, cond):
        """
        add a StoppingCondition, can be chained
        """
        self.stoppingConditions.append(cond)
        return self

    def withInitialTask(self, task, lvl):
        """
        initialize the queue with the given task at the given level, can be chained
        """
        self.queue = AcquisitionPriorityQueue()
        self.queue.put(task, lvl)
        return self
    
class DefaultNameHandler():
    """
    file name handler
    """
    
    def __init__(self, path, levels, prefix=None, ending = '.msr'):
        self.path = path
        self.levels = levels
        self.ending = ending
        if prefix is None:
            hash_object = hashlib.md5(bytes(str(time()), "utf-8"))
            hex_dig = hash_object.hexdigest()
            self.prefix = str(hex_dig)
        else:
            self.prefix = prefix
            
        if not os.path.exists(path):
            os.makedirs(path)
            
    def _mkdir_if_necessary(self):
        pass
            
    def get_filename(self, idxes):
        insert = chain.from_iterable(zip([l.name for l in self.levels.levels[0:len(idxes)]], idxes))
        insert = list(insert)
        return ((self.prefix + '_{}_{}' * len(idxes)).format(*insert) + self.ending)
    
    def get_path(self, idxes):
        return os.path.join(self.path, self.get_filename(idxes))
