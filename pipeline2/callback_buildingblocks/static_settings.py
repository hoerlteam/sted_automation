from pathlib import Path
import json
from functools import reduce
from itertools import cycle
from operator import add
from typing import Sequence

from pipeline2.utils.dict_utils import generate_nested_dict, remove_path_from_dict, merge_dicts
from pipeline2.utils.parameter_constants import FOV_LENGTH_PARAMETERS, PIXEL_SIZE_PARAMETERS


class FOVSettingsGenerator:
    """
    SettingsGenerator to set field of view (FOV) to defined length and pixel size. 

    Parameters
    ----------
    lengths : iterable of 3d-vectors
        lengths of the FOVs to image
    pixel_sizes : iterable of 3d-vectors
        pixel sizes of FOVs to image
    as_measurements: boolean
        if more than one FOV is specified: whether to create multiple `measurements` or
        multiple `configurations` in one measurement
    """
    def __init__(self, lengths, pixel_sizes, as_measurements=True):

        # check if parameters are list of lists (or None), wrap single list
        if lengths is not None and not isinstance(lengths[0], Sequence):
            lengths = [lengths]
        if pixel_sizes is not None and not isinstance(pixel_sizes[0], Sequence):
            pixel_sizes = [pixel_sizes]

        self.lengths = lengths
        self.pixel_sizes = pixel_sizes
        self.as_measurements = as_measurements

    paths_fov_len = FOV_LENGTH_PARAMETERS
    paths_pixel_size = PIXEL_SIZE_PARAMETERS

    def __call__(self):

        # if neither lengths nor pixel sizes are given, return single empty update
        if (self.lengths is None) and (self.pixel_sizes is None):
            return [[({}, {})]]

        res = []

        lengths = self.lengths
        pixel_sizes = self.pixel_sizes

        if self.lengths is None:
            lengths = [[None] * len(self.pixel_sizes)]
        if self.pixel_sizes is None:
            pixel_sizes = [[None] * len(self.lengths)]

        for l, psz in zip(lengths, pixel_sizes):
            res_measurement_i = {}
            paths = cycle(zip(self.paths_fov_len, self.paths_pixel_size))
            for l_i, psz_i in zip(l, psz):
                path_l, path_psz = next(paths)
                if l_i is not None:
                    res_measurement_i = merge_dicts(res_measurement_i, generate_nested_dict(l_i, path_l))
                if psz_i is not None:
                    res_measurement_i = merge_dicts(res_measurement_i, generate_nested_dict(psz_i, path_psz))
            res.append([(res_measurement_i, {})])

        # return either list of single element lists of parameter pairs (multiple measurements @ 1 configuration)
        # or a list containing one list of all pairs (1 measurement @ multiple configurations)
        if self.as_measurements:
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
    pixel_sizes : iterable of 3d-vectors
        pixel sizes of FOVs to image
    first_lengths: iterable of 3d-vectors, optional
        lengths of the first FOV
    as_measurements: boolean
        if more than one FOV is specified: whether to create multiple `measurements` or
        multiple `configurations` in one measurement
    """

    def __init__(self, lengths, pixel_sizes, first_lengths=None, as_measurements=True):
        self.first_measurement = True
        super().__init__(lengths, pixel_sizes, as_measurements)
        self.first_lengths = self.lengths if first_lengths is None else first_lengths

    def __call__(self):
        if self.first_measurement:
            lens_temp = self.lengths
            self.lengths = self.first_lengths
        res = super().__call__()
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
    as_measurements: boolean
        if more than one FOV is specified: whether to create multiple `measurements` or
        multiple `configurations` in one measurement    
    """

    def __init__(self, modes, as_measurements=True):
        self.modes = modes
        self.as_measurements = as_measurements

    def __call__(self):
        res = []

        for mode in self.modes:
            resD = {}
            resD = merge_dicts(resD, generate_nested_dict(ScanModeSettingsGenerator.gen_mode_flag(mode), self._path))

            resD = merge_dicts(
                resD,
                generate_nested_dict(['ExpControl {}'.format(mode[i].upper()) if i < len(mode) else "None" for i in range(4)],
                                     self._path_axes))

            # z-cut -> sync line
            # FIXME: this causes weird problems in xz cut followed by any other image
            # therefore, we removed it for the time being...
            '''
            if len(mode) == 2 and 'z' in mode.lower():
                resD = update_dicts(resD, gen_json(1, 'Measurement/axes/num_synced'))
            '''
            res.append([(resD, {})])

        if self.as_measurements:
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


class JSONSettingsLoader:
    """
    load settings from JSON
    """

    # parameters to remove from measurement parameter dicts because they caused problems
    parameters_to_drop = [
        '/Measurement/LoopMeasurement', '/Measurement/ResumeIdx', # remove, otherwise Imspector complains that those parameters do not exist (yet?)
        '/Measurement/propset_id', # remove, otherwise we will always use a set propset
        ]

    def __init__(self, measurement_config_sources, hardware_config_sources=None, as_measurements=True):
        self.measurement_configs = []
        self.as_measurements = as_measurements

        # allow single config -> wrap in list
        if isinstance(measurement_config_sources, (str, Path, dict)):
            measurement_config_sources = [measurement_config_sources]

        for measurement_config_source in measurement_config_sources:
            if isinstance(measurement_config_source, (str, Path)):
                with open(measurement_config_source, 'r') as fd:
                    measurement_config_loaded = json.load(fd)
            elif isinstance(measurement_config_source, dict):
                measurement_config_loaded = measurement_config_source
            else:
                raise ValueError('configuration should be either a valid filename or an already loaded dict')

            # remove parameters known to cause problems
            for parameter_to_drop in self.parameters_to_drop:
                measurement_config_loaded = remove_path_from_dict(measurement_config_loaded, parameter_to_drop)

            self.measurement_configs.append(measurement_config_loaded)

        self.hardware_configs = []

        # option 1: no hardware configs provided, leave empty
        if hardware_config_sources is None:
            for _ in range(len(self.measurement_configs)):
                self.hardware_configs.append(dict())

        # option 2: load hardware configs
        else:

            if isinstance(hardware_config_sources, (str, Path, dict)):
                hardware_config_sources = [hardware_config_sources]

            if len(hardware_config_sources) != len(self.measurement_configs):
                raise ValueError('length of settings and measurement configs do not match')

            for hardware_config_source in hardware_config_sources:
                if isinstance(hardware_config_source, (str, Path)):
                    with open(hardware_config_source, 'r') as fd:
                        hardware_config_loaded = json.load(fd)
                elif isinstance(hardware_config_source, dict):
                    hardware_config_loaded = hardware_config_source
                else:
                    raise ValueError('configuration should be either a valid filename or an already loaded dict')

                self.hardware_configs.append(hardware_config_loaded)

    def __call__(self):
        res = []
        if self.as_measurements:
            for i in range(len(self.measurement_configs)):
                res.append([(self.measurement_configs[i], self.hardware_configs[i])])
        else:
            res_inner = []
            for i in range(len(self.measurement_configs)):
                res_inner.append((self.measurement_configs[i], self.hardware_configs[i]))
            res.append(res_inner)
        return res
