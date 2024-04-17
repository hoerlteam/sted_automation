import pprint
from operator import add
from functools import reduce
from itertools import cycle

from pipeline2.utils.dict_utils import update_dicts, generate_recursive_dict
from pipeline2.utils.parameter_constants import OFFSET_SCAN_PARAMETERS, FOV_LENGTH_PARAMETERS


class ValuesToSettingsDictCallback:

    """
    General callback to wrap other callbacks that return parameter values / collections of values
    and create microscope-/Imspector-compatible nested dicts of settings
    """
    def __init__(self, value_generator_callback, settings_paths, as_measurements=True, nested_generator_callback=False, hardware_settings=False):

        if len(settings_paths) == 0:
            raise ValueError("No settings path provided")

        self.settings_paths = settings_paths
        self.value_generator_callback = value_generator_callback

        # check if we have multiple lists/tuples of settings (instead of just one list of str)
        self.nested_settings = isinstance(settings_paths[0], (tuple, list))

        self.nested_generator_callback = nested_generator_callback
        self.hardware_settings = hardware_settings
        self.as_measurements = as_measurements

    @staticmethod
    def values_to_settings_dict(values, settings_paths, nested_settings=False):
        """
        combine a sequence of values and a sequence of setting paths into a nested settings dict
        (or optionally nested sequences of sequences of values and setting paths)
        """

        # even if we have just one set of values and settings, fake multiple "groups" of values and settings
        if not nested_settings:
            values = [values]
            settings_paths = [settings_paths]

        print(values)

        result_dict = {}
        # go through two layers: groups of settings, paths and the individual value, path in them
        for values_group, settings_paths_group in zip(values, settings_paths):
            for value, settings_path in zip(values_group, settings_paths_group):
                # components of value may be None, e.g. if we only want to update xy/z coordinates
                # and give the others as None: [None, None, z_coordinate]
                if value is None:
                    continue
                # merge into result dict
                result_dict = update_dicts(result_dict, generate_recursive_dict(value, settings_path))
        return result_dict

    def __call__(self):

        values = self.value_generator_callback()

        # we DON'T have a callback that returns multiple "configurations"
        # e.g., just a collection of coordinates (list of vectors) or ROIs (list of (coord, size) pairs)
        # wrap everything in a tuple so we can treat it the same as a collection of collections of parameter values
        if not self.nested_generator_callback:
            values = [(value,) for value in values]

        print(values)
        settings_for_measurement = []
        for values_measurement in values:
            settings_for_configuration = []
            for values_configuration in values_measurement:
                settings_dict = self.values_to_settings_dict(values_configuration, self.settings_paths, self.nested_settings)
                # settings for one configuration are tuple (measurement settings, hardware settings)
                # pick one, leave the other as empty dict
                settings_for_configuration.append(({}, settings_dict) if self.hardware_settings else (settings_dict, {}))
            settings_for_measurement.append(settings_for_configuration)

        # return list of measurements (default)
        if self.as_measurements:
            return settings_for_measurement
        # flatten into one measurement with many configurations
        else:
            return [reduce(add, settings_for_measurement)]




class DefaultScanOffsetsSettingsGenerator:
    offset_settings_paths = OFFSET_SCAN_PARAMETERS

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
            path = cycle(self.offset_settings_paths)
            for l in loc:
                p =  next(path)

                # components of loc may be None, e.g. if we only want to update z
                if l is None:
                    continue
                resD = update_dicts(resD, generate_recursive_dict(l, p))
            res.append([(resD, {})])

        # return updates as separate measurements -> list of size-1 lists (each containing )
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
            path = cycle(self.offset_settings_paths)
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
    offset_settings_paths = ['ExpControl/scan/range/coarse_x/g_off',
              'ExpControl/scan/range/coarse_y/g_off',
              'ExpControl/scan/range/coarse_z/g_off']


class ZDCOffsetSettingsGenerator(DefaultScanOffsetsSettingsGenerator):
    # TODO: check if this is still the correct path
    offset_settings_paths = ['ExpControl/scan/range/x/off',
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

    def __init__(self, fieldGenerator, asMeasurements=True, fun=None):

        self.fieldGenerator = fieldGenerator
        self.asMeasurements = asMeasurements

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
            path = cycle(zip(self._paths_off, self._paths_len))
            for off, fov_len in zip(loc, fov):
                po, pl =  next(path)

                # components of loc may be None, e.g. if we only want to update z
                if off is not None:
                    resD = update_dicts(resD, generate_recursive_dict(off, po))
                if fov_len is not None:
                    resD = update_dicts(resD, generate_recursive_dict(fov_len, pl))
            res.append([(resD, {})])

        if self.asMeasurements:
            return res
        else:
            return [reduce(add, res)]


if __name__ == '__main__':
    values = [[[1,2,3], [3,4,5]], [[1,2,3], [3,4,5]]]
    value_callback = lambda: values
    gen = ValuesToSettingsDictCallback(value_callback, (OFFSET_SCAN_PARAMETERS, FOV_LENGTH_PARAMETERS), nested_generator_callback=False, as_measurements=False, hardware_settings=True)
    res = gen()
    pprint.pprint(res)