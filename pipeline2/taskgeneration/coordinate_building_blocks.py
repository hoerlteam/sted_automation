from operator import add
from functools import reduce
from itertools import cycle

from ..utils.dict_utils import update_dicts, generate_recursive_dict


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
        #print('DEBUG', locs)

        res = []
        for loc in locs:
            resD = {}
            path = cycle(self._paths)
            for l in loc:
                p =  next(path)

                # components of loc may be None, e.g. if we only want to update z
                if l is None:
                    continue
                resD = update_dicts(resD, generate_recursive_dict(l, p))
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

                # components of loc may be None, e.g. if we only want to update z
                if l1 is None or l2 is None:
                    continue
                resD1 = update_dicts(resD1, generate_recursive_dict(l1, p))
                resD2 = update_dicts(resD2, generate_recursive_dict(l2, p))
            res.append([(resD1, {})] * self.repeat_channel + [(resD2, {})] * self.repeat_channel)
        if self.asMeasurements:
            return res
        else:
            return [reduce(add, res)]


class DefaultStageOffsetsSettingsGenerator(DefaultScanOffsetsSettingsGenerator):
    _paths = ['ExpControl/scan/range/coarse_x/g_off',
              'ExpControl/scan/range/coarse_y/g_off',
              'ExpControl/scan/range/coarse_z/g_off']


class ZDCOffsetSettingsGenerator(DefaultScanOffsetsSettingsGenerator):
    _paths = ['ExpControl/scan/range/x/off',
              'ExpControl/scan/range/y/off',
              'ExpControl/scan/range/offsets/coarse/z/g_off'
              ]


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

    def __init__(self, fieldGenerator, pszs=None, asMeasurements=True, fun=None):

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

            pszs_t = [None] * len(loc) if self.pszs is None else self.pszs

            resD = {}
            path = cycle(zip(self._paths_off, self._paths_psz, self._paths_len))
            for off, psz, fov_len in zip(loc, pszs_t, fov):
                po, ppsz, pl =  next(path)

                # components of loc may be None, e.g. if we only want to update z
                if off is not None:
                    resD = update_dicts(resD, generate_recursive_dict(off, po))
                if psz is not None:
                    resD = update_dicts(resD, generate_recursive_dict(psz, ppsz))
                if fov_len is not None:
                    resD = update_dicts(resD, generate_recursive_dict(fov_len, pl))
            res.append([(resD, {})])

        if self.asMeasurements:
            return res
        else:
            return [reduce(add, res)]
