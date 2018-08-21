from itertools import cycle
from copy import deepcopy
from pipeline2.util import remove_filter_from_dict, gen_json, update_dicts, filter_dict
from operator import add
from functools import reduce
import json

from .fov_util import group_in_bounding_boxes

from unittest.mock import MagicMock


def _relative_spiral_generator(steps, start=[0,0]):
    """
    generator for regular spiral coordinates around a starting point
    with given step sizes
    """
    n = 0
    yield start[0:2].copy()
    while True:
        bookmark = [- n * steps[0] + start[0], n * steps[1] + start[1]]
        for _ in range(2*n):
            yield bookmark.copy()
            bookmark[0] += steps[0]
        for _ in range(2*n):
            yield bookmark.copy()
            bookmark[1] -= steps[1]
        for _ in range(2*n):
            yield bookmark.copy()
            bookmark[0] -= steps[0]
        for _ in range(2*n):
            yield bookmark.copy()
            bookmark[1] += steps[1]
        n += 1


class AcquisitionTaskGenerator():
    def __init__(self, level, *updateGens):

        self.level = level
        self.updateGens = updateGens
        self.delay = 0
        self.taskFilters = []

    def withDelay(self, delay):
        """
        a delay that will be added to every generated task
        (e.g. to wait for the stage to move)
        """
        self.delay = delay
        return self

    def withFilters(self, *filters):
        for filter in filters:
            self.taskFilters.append(filter)
        return self

    def __call__(self, pipeline):
        # broadcast meausurement updates ((u1, u2), (u3) -> ((u1, u3), (u2, u3)))

        updates = [updateGenI() for updateGenI in self.updateGens]

        maxMeasurements = max((len(updateI) for updateI in updates))
        minMeasurements = min((len(updateI) for updateI in updates))

        cyclesMeas = [cycle(updateI) for updateI in updates]

        # nothing to do, e.g. no detections
        if minMeasurements == 0:
            return

        # update the filters
        for filt in self.taskFilters:
            filt.update()

        for _ in range(maxMeasurements):

            # broadcast configurations within measurements
            # FIXME: why StopIteration here?
            configs = [next(meas) for meas in cyclesMeas]
            maxConfigs = max((len(confI) for confI in configs))
            cyclesConfig = [cycle(confI) for confI in configs]

            finalConfs = []

            for _ in range(maxConfigs):
                finalConfs.append([next(upd) for upd in cyclesConfig])

            # print(json.dumps(finalConfs, indent=2))
            task = AcquisitionTask(self.level).withUpdates(finalConfs).withDelay(self.delay)

            # reject task if it does not conform to a filter
            skip = False
            for filt in self.taskFilters:
                if not filt.conforms(task):
                    skip = True
            if skip:
                continue


            pipeline.queue.put(task, self.level)


class AcquisitionTask():
    """
    a dummy acquisition task, that will repeat itself every second
    """

    def __init__(self, pipelineLevel):
        self.pipelineLevel = pipelineLevel
        self.measurementUpdates = []
        self.settingsUpdates = []
        self.delay = 0

    def withUpdates(self, updates):
        for u in updates:
            measUpdates = [m for m, s in u]
            settingsUpdates = [s for m, s in u]
            self.measurementUpdates.append(measUpdates)
            self.settingsUpdates.append(settingsUpdates)
        return self

    def withDelay(self, delay=0):
        self.delay = delay
        return self

    @property
    def numAcquisitions(self):
        return len(self.measurementUpdates)

    def getUpdates(self, n):
        return self.measurementUpdates[n], self.settingsUpdates[n]

    def getAllUpdates(self):
        return [self.getUpdates(n) for n in range(self.numAcquisitions)]

    # TODO: move, this should become part of an analysis callback
    def __call__(self, pipeline, *args, **kwargs):
        print('pipeline {}: do dummy acquisition on level {}'.format(pipeline.name, self.pipelineLevel))
        # sleep(1)
        pipeline.queue.put(AcquisitionTask(self.pipelineLevel).withDelay(self.delay), self.pipelineLevel)


class NewestDataSelector():
    """

    """

    def __init__(self, pipeline, level):
        self.pipeline = pipeline
        self.lvl = level

    def get_data(self):
        # create index of measurement (indices of all levels until lvl)
        latestMeasurementIdx = tuple([self.pipeline.counters[l] for l in self.pipeline.pipelineLevels.levels[
                                                                         0:self.pipeline.pipelineLevels.levels.index(
                                                                             self.lvl) + 1]])
        return self.pipeline.data.get(latestMeasurementIdx, None)


class NewestSettingsSelector():
    def __init__(self, pipeline, level):
        self.level = level
        self.pipeline = pipeline

    def __call__(self):
        pipeline = self.pipeline
        latestMeasurementIdx = tuple([pipeline.counters[l] for l in pipeline.pipelineLevels.levels[
                                                                    0:pipeline.pipelineLevels.levels.index(
                                                                        self.level) + 1]])
        data = pipeline.data.get(latestMeasurementIdx, None)
        return [list(zip(data.measurementSettings, data.globalSettings))]


class BoundingBoxLocationGrouper():
    """
    Wrapper for a locationGenerator that groups locations into bounding boxes of defined size.
    This may be necessary to avoid multiple imaging of the same object.

    Parameters
    ----------
    locationGenerator : object implementing `get_locations` (returning iterable of 3d location vectors)
        generator of locations
    boundingBoxSize : 3d vector (array-like)
        size of the bounding boxes to group in (same unit as vecors returned by locationGenerator)
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
        for l, psz in zip(self.lengths, self.pixelSizes ):
            resD = {}
            paths = cycle(zip(self._paths_len, self._paths_psz))
            for l_i, psz_i in zip(l, psz):
                path_l, path_psz = next(paths)
                resD = update_dicts(resD, gen_json(l_i, path_l))
                resD = update_dicts(resD, gen_json(psz_i, path_psz))
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
        res = super().__call__()
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
            resD = update_dicts(resD, gen_json(DefaultScanModeSettingsGenerator.gen_mode_flag(mode), self._path))
            
            resD = update_dicts(
                resD,
                gen_json(['ExpControl {}'.format(mode[i].upper()) if i < len(mode) else "None" for i in range(4)],
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
        'x' : 0,
        'y' : 1,
        'z' : 2,
        't' : 3
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

class DefaultScanOffsetsSettingsGenerator():
    _paths = ['ExpControl/scan/range/x/off',
              'ExpControl/scan/range/y/off',
              'ExpControl/scan/range/z/off'
              ]

    def __init__(self, locationGenerator, asMeasurements=True, fun=None):
        self.locationGenerator = locationGenerator
        self.asMeasurements = asMeasurements
        if fun is None:
            self.fun = locationGenerator.get_locations
        else:
            self.fun = fun

    def __call__(self):
        '''
        Returns
        -------
        settings: list of list of (measurement_parameters, global_parameters) tuples
            parameter updates (global updates == {}) for every configuration in every measurement to acquire.
        '''
        locs = self.fun()

        res = []
        for loc in locs:
            resD = {}
            path = cycle(self._paths)
            for l in loc:
                p =  next(path)

                # components of loc may be noe, e.g. if we only want to update z
                if l is None:
                    continue
                resD = update_dicts(resD, gen_json(l, p))
            res.append([(resD, {})])
        if self.asMeasurements:
            return res
        else:
            return [reduce(add, res)]


class PairedDefaultScanOffsetsSettingsGenerator(DefaultScanOffsetsSettingsGenerator):
    def __init__(self, locationGenerator, asMeasurements=True, fun=None, repeat_channel=1):
        self.repeat_channel = repeat_channel
        super().__init__(locationGenerator, asMeasurements, fun)

    def __call__(self):
        locs = self.fun()

        res = []
        for loc1, loc2 in locs:
            resD1 = {}
            resD2 = {}
            path = cycle(self._paths)
            for l1, l2 in zip(loc1, loc2):
                p = next(path)

                # components of loc may be noe, e.g. if we only want to update z
                if l1 is None or l2 is None:
                    continue
                resD1 = update_dicts(resD1, gen_json(l1, p))
                resD2 = update_dicts(resD2, gen_json(l2, p))
            res.append([(resD1, {})] * self.repeat_channel + [(resD2, {})] * self.repeat_channel)
        if self.asMeasurements:
            return res
        else:
            return [reduce(add, res)]



class DefaultScanFieldSettingsGenerator():

    _paths_off = ['ExpControl/scan/range/x/off',
                  'ExpControl/scan/range/y/off',
                  'ExpControl/scan/range/z/off'
                  ]

    _paths_len = ['ExpControl/scan/range/x/len',
                  'ExpControl/scan/range/y/len',
                  'ExpControl/scan/range/z/len'
                  ]
    _paths_psz = ['ExpControl/scan/range/x/psz',
                  'ExpControl/scan/range/y/psz',
                  'ExpControl/scan/range/z/psz'
                  ]

    def __init__(self, fieldGenerator, pszs, asMeasurements=True, fun=None):

        self.fieldGenerator = fieldGenerator
        self.asMeasurements = asMeasurements
        self.pszs = pszs

        if fun is None:
            self.fun = fieldGenerator.get_fields
        else:
            self.fun = fun

    def __call__(self):
        '''
        Returns
        -------
        settings: list of list of (measurement_parameters, global_parameters) tuples
            parameter updates (global updates == {}) for every configuration in every measurement to acquire.
        '''
        fields = self.fun()

        res = []
        for loc, fov in fields:
            resD = {}
            path = cycle(zip(self._paths_off, self._paths_psz, self._paths_len))
            for off, psz, fov_len in zip(loc, self.pszs, fov):
                po, ppsz, pl =  next(path)

                # components of loc may be None, e.g. if we only want to update z
                if off is not None:
                    resD = update_dicts(resD, gen_json(off, po))
                if psz is not None:
                    resD = update_dicts(resD, gen_json(psz, ppsz))
                if fov_len is not None:
                    resD = update_dicts(resD, gen_json(fov_len, pl))
            res.append([(resD, {})])

        if self.asMeasurements:
            return res
        else:
            return [reduce(add, res)]


class DefaultStageOffsetsSettingsGenerator(DefaultScanOffsetsSettingsGenerator):
    _paths = ['ExpControl/scan/range/offsets/coarse/x/g_off',
              'ExpControl/scan/range/offsets/coarse/y/g_off',
              'ExpControl/scan/range/offsets/coarse/z/g_off']

class ZDCOffsetSettingsGenerator(DefaultScanOffsetsSettingsGenerator):
    _paths = ['ExpControl/scan/range/x/off',
              'ExpControl/scan/range/y/off',
              'ExpControl/scan/range/offsets/coarse/z/g_off'
              ]


class DefaultLocationKeeper():
    """
    this wrapper can be used to keep just the location-related updates from the
    output of a settings generator.
    """

    _filtersToKeep = ['ExpControl/scan/range/offsets',
                        'ExpControl/scan/range/x/off',
                        'ExpControl/scan/range/x/g_off',
                        'ExpControl/scan/range/y/off',
                        'ExpControl/scan/range/y/g_off',
                        'ExpControl/scan/range/z/off',
                        'ExpControl/scan/range/z/g_off'
                      ]

    def __init__(self, coordinateProvider):
        self.coordinateProvider = coordinateProvider

    def __call__(self):
        res = []
        for l in self.coordinateProvider():
            lModified = []
            for meas, settings in l:
                measI = {}
                for f in DefaultLocationKeeper._filtersToKeep:
                    mI = filter_dict(meas, f, False)
                    measI = update_dicts(measI, gen_json(mI, f) if not (mI is None) else {})
                lModified.append((measI, settings))
            res.append(lModified)
        return res

class DefaultLocationRemover():
    """
    this wrapper can be used to remove location-related updates from the output
    of a settings generator.
    if will remove the corresponding settings from every measurement dict
    and leave the rest as-is.
    """

    _filtersToRemove = ['ExpControl/scan/range/offsets',
                        'ExpControl/scan/range/x/off',
                        'ExpControl/scan/range/x/g_off',
                        'ExpControl/scan/range/y/off',
                        'ExpControl/scan/range/y/g_off',
                        'ExpControl/scan/range/z/off',
                        'ExpControl/scan/range/z/g_off',
                        'OlympusIX/stage',
                        'OlympusIX/scanrange']

    def __init__(self, coordinateProvider):
        self.coordinateProvider = coordinateProvider

    def __call__(self):
        res = []
        for l in self.coordinateProvider():
            lModified = []
            for meas, settings in l:
                measI = deepcopy(meas)
                for f in DefaultLocationRemover._filtersToRemove:
                    measI = remove_filter_from_dict(measI, f)
                    if measI is None:
                        measI = {}
                lModified.append((measI, settings))

            res.append(lModified)
        return res

class SpiralOffsetGenerator():
    def __init__(self):
        self.fov = [5e-5, 5e-5]
        self.start = [0, 0]
        self.zOff = None
        self.gen = _relative_spiral_generator(self.fov, self.start)
        self.verbose = False
    def withFOV(self, fov):
        self.fov = fov
        self.gen = _relative_spiral_generator(self.fov, self.start)
        return self
    def withStart(self, start):
        self.start = start
        self.gen = _relative_spiral_generator(self.fov, self.start)
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
    def __init__(self, measurementConfigFileNames, settingsConfigFileNames=None, asMeasurements=True):
        self.measConfigs = []
        self.asMeasurements = asMeasurements
        for mFile in measurementConfigFileNames:
            with open(mFile, 'r') as fd:
                d = json.load(fd)
                # remove, otherwise Imspector complains that those parameters do not exist (yet?)
                d = remove_filter_from_dict(d, '/Measurement/LoopMeasurement')
                d = remove_filter_from_dict(d, '/Measurement/ResumeIdx')
                # remove, otherwise we will always use a set propset
                d = remove_filter_from_dict(d, '/Measurement/propset_id')
                self.measConfigs.append(d)
        
        self.settingsConfigs = []
        if settingsConfigFileNames is None:
            for _ in range(len(self.measConfigs)):
                self.settingsConfigs.append(dict())
        else:
            if len(settingsConfigFileNames) != len(self.measConfigs):
                raise ValueError('length of settings and measurement configs dont match')
            for sFile in settingsConfigFileNames:
                with open(sFile, 'r') as fd:
                    d = json.load(fd)
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
    
def main():

    from pipeline2.taskgeneration import SpiralOffsetGenerator
    spiralGen = SpiralOffsetGenerator().withStart([0,0]).withFOV([5,5]).withZOffset(1)
    for _ in range(5):
        print(spiralGen.get_locations())


def ATGTest():
    locMock = MagicMock(return_value = [])
    locMock.get_locations = MagicMock(return_value = [])
    og = ZDCOffsetSettingsGenerator(locMock)

    pipelineMock = MagicMock()
    atg = AcquisitionTaskGenerator(0, og)
    atg(pipelineMock)


    print(locMock.get_locations())

if __name__ == '__main__':
    ATGTest()
