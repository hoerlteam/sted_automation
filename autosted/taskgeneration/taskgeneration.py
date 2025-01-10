from itertools import cycle
from typing import Sequence

from autosted.utils.dict_utils import merge_dicts


class AcquisitionTaskGenerator:
    def __init__(self, level, *update_generators, delay=0):

        self.level = level
        # ignore update generators that are None
        # that way, we can safely pass None to constructor
        # e.g. when no parameter change is desired by the user
        self.update_generators = [u for u in update_generators if u is not None]
        self.delay = delay
        self.task_filters = []

    def add_task_filters(self, *filters):
        for filter_i in filters:
            self.task_filters.append(filter_i)
        return self

    def __call__(self):
        """
        Run wrapped callbacks and broadcast and combine measurement updates,
        repeating single updates from one callback to match with multiple from another:
        e.g.: (u1, u2, u3), (u4) -> (u1 + u4), (u2 + u4), (u3 + u4)

        If multiple callbacks return more than one measurement, they must have the same number:
        e.g.: (u1, u2), (u3, u4) -> (u1 + u3), (u2 + u4)

        If any callback returns an empty list of updates, nothing will be enqueued.
        """

        # run all wrapped callbacks, they should each return a list of updates
        # that should be combined into a list of measurements to enqueue
        updates = [update_generator() for update_generator in self.update_generators]

        # broadcast updates at the measurement level
        updates_per_measurements_broadcast = broadcast_updates(updates)

        tasks = []
        for measurement_update in updates_per_measurements_broadcast:

            # broadcast configurations within measurements
            config_updates = broadcast_updates(measurement_update)

            task = AcquisitionTask(self.level)
            task.add_updates(config_updates)
            task.delay = self.delay

            # reject task if it does not conform to a task_filter
            skip = False
            for task_filter in self.task_filters:
                if not task_filter.check(task, self.level):
                    skip = True
            if skip:
                continue

            tasks.append(task)

        return self.level, tasks


class AcquisitionTask:

    '''
    Wrapper for (measurement_settings, hardware_settings) dict-pairs
    representing a measurement (consisting of multiple configurations) to be run.
    '''

    # TODO: this has minimal added value to just a list of parameter pairs, remove?

    def __init__(self, pipeline_level):
        self.pipeline_level = pipeline_level
        self.measurement_updates = []
        self.hardware_updates = []
        self.delay = 0

    def add_updates(self, updates):
        for u in updates:
            measurement_updates = [m for m, s in u]
            hardware_updates = [s for m, s in u]
            self.measurement_updates.append(measurement_updates)
            self.hardware_updates.append(hardware_updates)

    @property
    def num_acquisitions(self):
        return len(self.measurement_updates)

    def __len__(self):
        return self.num_acquisitions

    def __getitem__(self, idx):
        return self.get_updates(idx, True)

    def get_updates(self, n, concatenate=False):
        hardware_updates = self.hardware_updates[n]
        measurement_updates = self.measurement_updates[n]
        if concatenate:
            hardware_updates = merge_dicts(*hardware_updates)
            measurement_updates = merge_dicts(*measurement_updates)
        return measurement_updates, hardware_updates

    def get_all_updates(self, concatenate=False):
        return [self.get_updates(n, concatenate) for n in range(self.num_acquisitions)]


def broadcast_updates(updates: Sequence[Sequence]):

    # get number of measurements / configurations in each update
    update_lengths = [len(update_i) for update_i in updates]

    # nothing to do, e.g. no detections
    if min(update_lengths) == 0:
        return []

    # run some sanity checks to make sure we can do reasonable broadcasting of updates
    update_lengths_unique = set(update_lengths)
    if len(update_lengths_unique) > 2:
        raise ValueError(f"Can't combine updates with more than two lengths: {update_lengths_unique}")
    # if we have two different lengths, it is okay only if one of them is 1
    # then, the single update will be combined with all other updates
    if len(update_lengths_unique) == 2 and 1 not in update_lengths_unique:
        raise ValueError(f"Can't combine updates with two different lengths > 1: {update_lengths_unique}")

    # get combined updates: cycle so ones with len < maximum len will be repeated
    result = []
    cycle_updates = [cycle(update_i) for update_i in updates]
    for _ in range(max(update_lengths_unique)):
        result.append(tuple(next(cycle_i) for cycle_i in cycle_updates))

    # return as tuple
    return tuple(result)


def main():

    from autosted.callback_buildingblocks.regular_position_generators import SpiralOffsetGenerator
    spiralGen = SpiralOffsetGenerator().withStart([0,0]).withFOV([5,5]).withZOffset(1)
    for _ in range(5):
        print(spiralGen.get_locations())


def ATGTest():
    from unittest.mock import MagicMock
    from autosted.callback_buildingblocks.coordinate_value_wrappers import ZDCOffsetSettingsGenerator

    locMock = MagicMock(return_value = [])
    locMock.get_locations = MagicMock(return_value = [])
    og = ZDCOffsetSettingsGenerator(locMock)

    pipelineMock = MagicMock()
    atg = AcquisitionTaskGenerator(0, og)
    atg(pipelineMock)


    print(locMock.get_locations())


if __name__ == '__main__':
    test_updates = [('u1', 'u2', 'u3'), ['v1', 'v2', 'v3']]
    print(broadcast_updates(test_updates))

    u1 = ['coords1', 'coords1-1']
    u2 = ['coords2', 'coords2-1']
    u3 = ['coords3', 'coords3-1']
    v1 = ['settings1', 'settings2',]
    test_updates = ((u1, u2, u3), (v1,))

    for meas_updates in broadcast_updates(test_updates):
        print(broadcast_updates(meas_updates))
