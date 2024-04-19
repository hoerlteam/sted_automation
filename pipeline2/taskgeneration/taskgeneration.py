from itertools import cycle
from operator import add
from functools import reduce
import json
from pathlib import Path
from unittest.mock import MagicMock
from typing import Sequence

from pipeline2.utils.dict_utils import update_dicts, remove_path_from_dict, generate_recursive_dict
from pipeline2.taskgeneration.fov_util import group_in_bounding_boxes
from pipeline2.utils.tiling import relative_spiral_generator


class AcquisitionTaskGenerator:
    def __init__(self, level, *update_generators, delay=0):

        self.level = level
        # ignore update generators that are None
        # that way, we can safely pass None to constructor
        # e.g. when no parameter change is desired by the user
        self.update_generators = [u for u in update_generators if u is not None]
        self.delay = delay
        self.taskFilters = []

    def add_task_filters(self, *filters):
        for filter in filters:
            self.taskFilters.append(filter)
        return self

    def __call__(self, pipeline):
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

        # update the filters
        # TODO: check if this can be simplified/removed?
        for task_filter in self.taskFilters:
            task_filter.update()

        tasks = []
        for measurement_update in updates_per_measurements_broadcast:

            # broadcast configurations within measurements
            config_updates = broadcast_updates(measurement_update)

            # print(json.dumps(finalConfs, indent=2))
            task = AcquisitionTask(self.level).withUpdates(config_updates).withDelay(self.delay)

            # reject task if it does not conform to a task_filter
            skip = False
            for task_filter in self.taskFilters:
                if not task_filter.conforms(task):
                    skip = True
            if skip:
                continue

            tasks.append(task)

        return self.level, tasks


class AcquisitionTask():
    """
    a dummy acquisition task, that will repeat itself every second
    """

    def __init__(self, pipeline_level):
        self.pipeline_level = pipeline_level
        self.measurement_updates = []
        self.hardware_updates = []
        self.delay = 0

    def withUpdates(self, updates):
        for u in updates:
            measUpdates = [m for m, s in u]
            settingsUpdates = [s for m, s in u]
            self.measurement_updates.append(measUpdates)
            self.hardware_updates.append(settingsUpdates)
        return self

    def withDelay(self, delay=0):
        self.delay = delay
        return self

    @property
    def numAcquisitions(self):
        return len(self.measurement_updates)

    def getUpdates(self, n):
        return self.measurement_updates[n], self.hardware_updates[n]

    def getAllUpdates(self):
        return [self.getUpdates(n) for n in range(self.numAcquisitions)]

    # TODO: move, this should become part of an analysis callback
    def __call__(self, pipeline, *args, **kwargs):
        print('pipeline {}: do dummy acquisition on level {}'.format(pipeline.name, self.pipeline_level))
        # sleep(1)
        pipeline.queue.put(AcquisitionTask(self.pipeline_level).withDelay(self.delay), self.pipeline_level)


class BoundingBoxLocationGrouper():
    """
    Wrapper for a locationGenerator that groups locations into bounding boxes of defined size.
    This may be necessary to avoid multiple imaging of the same object.

    Parameters
    ----------
    locationGenerator : object implementing `get_locations` (returning iterable of 3d location vectors)
        generator of locations
    boundingBoxSize : 3d vector (array-like)
        size of the bounding boxes to group in (same unit as vectors returned by locationGenerator)
    """
    def __init__(self, locationGenerator, boundingBoxSize):
        self.locationGenerator = locationGenerator
        self.boundingBoxSize = boundingBoxSize
        self.verbose = False
        
    def withVerbose(self, verbose=True):
        self.verbose = verbose
        return self

    def get_locations(self):
        xs = self.locationGenerator.get_locations()
        res = group_in_bounding_boxes(xs, self.boundingBoxSize)
        
        if self.verbose:
            print('grouped detections into {} FOVS:'.format(len(res)))
            for loc in res:
                print(loc)
        return res


class LocalizationNumberFilter():
    """
    Wrapper for a locationGenerator that will discard all localizations, if there are too few or too many

    Parameters
    ----------
    locationGenerator : object implementing `get_locations`
        generator of locations
    min: int, optional
        minimum number of localizations
    max: int, optional
        maximum number of localizations
    """
    def __init__(self, locationGenerator, min=None, max=None):
        self.locationGenerator = locationGenerator
        self.min = min
        self.max = max
        self.verbose = False

    def withVerbos(self, verbose=True):
        self.verbose = verbose
        return self

    def get_locations(self):
        locs = self.locationGenerator.get_locations()
        n_locs = len(locs)

        # return all or nothing, depending on number of locs
        if self.min is not None and n_locs < self.min:
            return []
        elif self.max is not None and n_locs > self.max:
            return []
        else:
            return locs


class DefaultFOVSettingsGenerator():
    """
    SettingsGenerator to set field of view (FOV) to defined length and pixel size. 
    
    Parameters
    ----------
    lengths : iterable of 3d-vectors
        lengths of the FOVs to image
    pixelSizes : iterable of 3d-vectors
        pixel sizes of FOVs to image
    asMeasurements: boolean
        if more than one FOV is specified: whether to create multiple `measurements` or
        multiple `configurations` in one measurement
    """
    def __init__(self, lengths, pixelSizes, asMeasurements=True):
        self.lengths = lengths
        self.pixelSizes = pixelSizes
        self.asMeasurements = asMeasurements

    _paths_len = ['ExpControl/scan/range/x/len',
                  'ExpControl/scan/range/y/len',
                  'ExpControl/scan/range/z/len'
                  ]
    _paths_psz = ['ExpControl/scan/range/x/psz',
                  'ExpControl/scan/range/y/psz',
                  'ExpControl/scan/range/z/psz'
                  ]

    def __call__(self):

        res = []

        if (self.lengths is None) and (self.pixelSizes is None):
            return [[({},{})]]

        _lengths = self.lengths
        _pixelSizes = self.pixelSizes

        if self.lengths is None:
            _lengths = [None] * len(self.pixelSizes)

        if self.pixelSizes is None:
            _pixelSizes = [None] * len(self.lengths)

        for l, psz in zip(_lengths, _pixelSizes ):
            resD = {}
            paths = cycle(zip(self._paths_len, self._paths_psz))
            for l_i, psz_i in zip(l if l is not None else [None] * len(psz), psz if psz is not None else [None] * len(l)):
                path_l, path_psz = next(paths)
                if l_i is not None:
                    resD = update_dicts(resD, generate_recursive_dict(l_i, path_l))
                if psz_i is not None:
                    resD = update_dicts(resD, generate_recursive_dict(psz_i, path_psz))
            res.append([(resD, {})])
        if self.asMeasurements:
            return res
        else:
            return [reduce(add, res)]

        
class DifferentFirstFOVSettingsGenerator(DefaultFOVSettingsGenerator):
    """
    SettingsGenerator to set field of view (FOV) to defined length and pixel size.
    A separate length can be set for the first FOV (e.g. to scan a larger z stack
    after moving a large distance)
    
    Parameters
    ----------
    lengths : iterable of 3d-vectors
        lengths of the FOVs to image
    pixelSizes : iterable of 3d-vectors
        pixel sizes of FOVs to image
    firstLengths: iterable of 3d-vectors, optional
        lengths of the first FOV
    asMeasurements: boolean
        if more than one FOV is specified: whether to create multiple `measurements` or
        multiple `configurations` in one measurement
    """
    
    def __init__(self, lengths, pixelSizes, firstLengths=None, asMeasurements=True):
        self.first_measurement = True
        super().__init__(lengths, pixelSizes, asMeasurements)
        self.first_lengths = self.lengths if firstLengths is None else firstLengths
        
    def __call__(self):
        if self.first_measurement:
            lens_temp = self.lengths
            self.lengths = self.first_lengths
        res = super()()
        if self.first_measurement:
            self.lengths = lens_temp
            self.first_measurement = False
        return res    


class DefaultScanModeSettingsGenerator():
    """
    SettingsGenerator to set the scan mode (e.g xy, xyz, xy,...)
    
    Parameters
    ----------
    
    modes: iterable of strings
        the mode strings, e.g. 'xy'
    asMeasurements: boolean
        if more than one FOV is specified: whether to create multiple `measurements` or
        multiple `configurations` in one measurement    
    """
    
    def __init__(self, modes, asMeasurements=True):
        self.modes = modes
        self.asMeasurements = asMeasurements
        
    def __call__(self):
        res = []
        
        for mode in self.modes:
            resD = {}
            resD = update_dicts(resD, generate_recursive_dict(DefaultScanModeSettingsGenerator.gen_mode_flag(mode), self._path))
            
            resD = update_dicts(
                resD,
                generate_recursive_dict(['ExpControl {}'.format(mode[i].upper()) if i < len(mode) else "None" for i in range(4)],
                                        self._path_axes))
            
            # z-cut -> sync line
            # FIXME: this causes weird problems in xz cut followed by any other image
            # therefore, we removed it for the time being...
            '''
            if len(mode) == 2 and 'z' in mode.lower():
                resD = update_dicts(resD, gen_json(1, 'Measurement/axes/num_synced'))
            '''
            res.append([(resD, {})])
        
        if self.asMeasurements:
            return res
        else:
            return [reduce(add, res)]
        
    _path = 'ExpControl/scan/range/mode'
    _path_axes = 'Measurement/axes/scan_axes'
        
    @staticmethod
    def gen_mode_flag(mode_str):

        _mode_vals = {
            'x': 0,
            'y': 1,
            'z': 2,
            't': 3
        }

        if len(mode_str) > 4:
            return None
        res = 0
        for _ in range(3 - len(mode_str)):
            res = (res + 1) << 2
            res = (res + 1) << 2
        for i, c in enumerate(reversed(mode_str)):
            res = res << 2
            res = res + _mode_vals[c]
            if not i == len(mode_str) - 1:
                res = res << 2
        return res


class StagePositionListGenerator:

    # TODO: add possibility to reset index during acquisition?
    #  -> might be necessary to re-image same positions multiple times?

    def __init__(self, positions, verbose=False):
        self.positions = positions
        self.idx = 0
        self.verbose = verbose

    def get_all_locations(self):
        # return copy of all positions
        return list(self.positions)

    def get_locations(self):

        # no more positions to image at
        if self.idx >= len(self.positions):
            return []

        # get next position and increment index
        coords = self.positions[self.idx]
        self.idx += 1   

        if self.verbose:
            print(self.__class__.__name__ + ': new coordinates: ' + str(coords))

        return [coords]


class SpiralOffsetGenerator():
    def __init__(self):
        self.fov = [5e-5, 5e-5]
        self.start = [0, 0]
        self.zOff = None
        self.gen = relative_spiral_generator(self.fov, self.start)
        self.verbose = False
    def withFOV(self, fov):
        self.fov = fov
        self.gen = relative_spiral_generator(self.fov, self.start)
        return self
    def withStart(self, start):
        self.start = start
        self.gen = relative_spiral_generator(self.fov, self.start)
        return self

    def withZOffset(self, zOff):
        self.zOff = zOff
        return self
    def withVerbose(self, verbose=True):
        self.verbose = verbose
        return self

    def get_locations(self):
        res = next(self.gen) if (self.zOff is None) else (next(self.gen) + [self.zOff])
        if self.verbose:
            print(self.__class__.__name__ + ': new coordinates: ' + str(res))
        return [res]

class JSONFileConfigLoader():
    """
    load settings from JSON dump
    """

    # TODO: rename since we now also support using an already loaded config instead of loading from a file

    # parameters to remove from measurement parameter dicts because they caused problems
    parameters_to_drop = [
        '/Measurement/LoopMeasurement', '/Measurement/ResumeIdx', # remove, otherwise Imspector complains that those parameters do not exist (yet?)
        '/Measurement/propset_id', # remove, otherwise we will always use a set propset
        ]

    def __init__(self, measurementConfigFileNames, settingsConfigFileNames=None, asMeasurements=True):
        self.measConfigs = []
        self.asMeasurements = asMeasurements

        for mFile in measurementConfigFileNames:
            if isinstance(mFile, (str, Path)):
                with open(mFile, 'r') as fd:
                    d = json.load(fd)
            elif isinstance(mFile, dict):
                d = mFile
            else:
                raise ValueError('configuration should be either a valid filename or an already loaded dict')

            # remove parameters known to cause problems
            for parameter_to_drop in JSONFileConfigLoader.parameters_to_drop:
                d = remove_path_from_dict(d, parameter_to_drop)

            self.measConfigs.append(d)
        
        self.settingsConfigs = []

        # option 1: no hardware configs provided, leave empty
        if settingsConfigFileNames is None:
            for _ in range(len(self.measConfigs)):
                self.settingsConfigs.append(dict())

        # option 2: load hardware configs
        else:
            if len(settingsConfigFileNames) != len(self.measConfigs):
                raise ValueError('length of settings and measurement configs do not match')

            for sFile in settingsConfigFileNames:
                if isinstance(sFile, (str, Path)):
                    with open(sFile, 'r') as fd:
                        d = json.load(fd)
                elif isinstance(sFile, dict):
                    d = sFile
                else:
                    raise ValueError('configuration should be either a valid filename or an already loaded dict')

                self.settingsConfigs.append(d)
    
    def __call__(self):
        res = []
        if self.asMeasurements:
            for i in range(len(self.measConfigs)):
                res.append([(self.measConfigs[i], self.settingsConfigs[i])])
        else:
            resInner = []
            for i in range(len(self.measConfigs)):
                resInner.append((self.measConfigs[i], self.settingsConfigs[i]))
            res.append(resInner)
        return res


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

    from pipeline2.taskgeneration import SpiralOffsetGenerator
    spiralGen = SpiralOffsetGenerator().withStart([0,0]).withFOV([5,5]).withZOffset(1)
    for _ in range(5):
        print(spiralGen.get_locations())


def ATGTest():

    from pipeline2.taskgeneration.coordinate_building_blocks import ZDCOffsetSettingsGenerator

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
