from itertools import cycle
from copy import deepcopy
from pipeline2.util import remove_filter_from_dict, gen_json, update_dicts, filter_dict
from operator import add
from functools import reduce
import json


def _relative_spiral_generator(steps, start=[0,0]):
    """
    generator for regular spiral coordinates around a starting point
    with given step sizes
    """
    n = 0
    yield start[0:2].copy()
    while True:
        bookmark = [- n * steps[0] + start[0], n * steps[1] + start[0]]
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

    def withDelay(self, delay):
        """
        a delay that will be added to every generated task
        (e.g. to wait for the stage to move)
        """
        self.delay = delay
        return self

    def __call__(self, pipeline):
        # broadcast meausurement updates ((u1, u2), (u3) -> ((u1, u3), (u2, u3)))

        updates = [updateGenI() for updateGenI in self.updateGens]

        maxMeasurements = max((len(updateI) for updateI in updates))
        cyclesMeas = [cycle(updateI) for updateI in updates]

        if maxMeasurements == 0:
            return
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
            pipeline.queue.put(AcquisitionTask(self.level).withUpdates(finalConfs).withDelay(self.delay), self.level)


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
        locs = self.fun()

        res = []
        for loc in locs:
            resD = {}
            path = cycle(self._paths)
            for l in loc:
                resD = update_dicts(resD, gen_json(l, next(path)))
            res.append([(resD, {})])
        if self.asMeasurements:
            return res
        else:
            return [reduce(add, res)]


class DefaultStageOffsetsSettingsGenerator(DefaultScanOffsetsSettingsGenerator):
    _paths = ['ExpControl/scan/range/offsets/coarse/x/g_off',
              'ExpControl/scan/range/offsets/coarse/y/g_off',
              'ExpControl/scan/range/offsets/coarse/z/g_off']

class ZDCOffsetSettingsGenerator(DefaultStageOffsetsSettingsGenerator):
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

    def get_locations(self):
        return [next(self.gen) if (self.zOff is None) else (next(self.gen) + [self.zOff])]

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
                    self.settingsConfigs.append(json.load(fd))
    
    def __call__(self):
        res = []
        if self.asMeasurements:
            for i in range(len(self.measConfigs)):
                res.append([(self.measConfigs[i], self.settingsConfigs[i])])
        else:
            resInner = []
            for i in range(len(self.measConfigs)):
                resInner.append([(self.measConfigs[i], self.settingsConfigs[i])])
            res.append(resInner)
        return res
    
def main():

    from pipeline2.taskgeneration import SpiralOffsetGenerator
    spiralGen = SpiralOffsetGenerator().withStart([0,0]).withFOV([5,5]).withZOffset(1)
    for _ in range(5):
        print(spiralGen.get_locations())

if __name__ == '__main__':
    main()