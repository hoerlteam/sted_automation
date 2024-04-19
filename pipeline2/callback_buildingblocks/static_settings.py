from pathlib import Path
from pipeline2.utils.dict_utils import generate_recursive_dict, remove_path_from_dict, update_dicts

import json
from functools import reduce
from itertools import cycle
from operator import add


class FOVSettingsGenerator:
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


class DifferentFirstFOVSettingsGenerator(FOVSettingsGenerator):
    """
    SettingsGenerator to set field of view (FOV) to defined length and pixel size.
    A separate length can be set for the first FOV (e.g. to scan a larger z stack
    to find focus if multiple runs happen automatically after each other)

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


class ScanModeSettingsGenerator:
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
            resD = update_dicts(resD, generate_recursive_dict(ScanModeSettingsGenerator.gen_mode_flag(mode), self._path))

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


class JSONSettingsLoader():
    """
    load settings from JSON dump
    """

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
            for parameter_to_drop in JSONSettingsLoader.parameters_to_drop:
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