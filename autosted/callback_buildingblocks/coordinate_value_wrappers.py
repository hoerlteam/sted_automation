import pprint
from operator import add
from functools import reduce

from autosted.utils.dict_utils import merge_dicts, generate_nested_dict
from autosted.utils.parameter_constants import OFFSET_SCAN_PARAMETERS, OFFSET_STAGE_GLOBAL_PARAMETERS, FOV_LENGTH_PARAMETERS


class ValuesToSettingsDictCallback:

    """
    General callback to wrap other callbacks that return parameter values / collections of values
    and create microscope-/Imspector-compatible nested dicts of settings
    """
    def __init__(self, value_generator_callback, settings_paths, as_measurements=True, nested_generator_callback=False, hardware_settings=False):
        """
        Parameters
        ----------
        value_generator_callback : callable that should return a sequence of
                                   a) parameter values (sequence) of the same length as settings_paths
                                   b) tuple of parameter value sequences if settings_paths as a nested sequence
                                   c) sequences of a) or b) if nested_generator_callback is True
        settings_paths: sequence of str or sequence of sequence of str
        as_measurements: bool, whether to return list of individual measurement settings
            (that may have multiple configurations if nested_generator_callback==True) or flatten everything into configurations of one measurement
        nested_generator_callback: bool, whether to expect nested results from value_generator_callback
        hardware_settings: bool, whether to make results hardware or measurement settings
        """

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

        result_dict = {}
        # go through two layers: groups of settings, paths and the individual value, path in them
        for values_group, settings_paths_group in zip(values, settings_paths):
            for value, settings_path in zip(values_group, settings_paths_group):
                # components of value may be None, e.g. if we only want to update xy/z coordinates
                # and give the others as None: [z_coordinate, None, None]
                if value is None:
                    continue
                # merge into result dict
                result_dict = merge_dicts(result_dict, generate_nested_dict(value, settings_path))
        return result_dict

    def __call__(self):

        values = self.value_generator_callback()

        # we DON'T have a callback that returns multiple "configurations"
        # e.g., just a collection of coordinates (list of vectors) or ROIs (list of (coord, size) pairs)
        # wrap everything in a tuple so we can treat it the same as a collection of collections of parameter values
        if not self.nested_generator_callback:
            values = [(value,) for value in values]

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


class ScanOffsetsSettingsGenerator(ValuesToSettingsDictCallback):
    def __init__(self, location_generator, as_measurements=True):
        super().__init__(location_generator, OFFSET_SCAN_PARAMETERS, as_measurements)


class StageOffsetsSettingsGenerator(ValuesToSettingsDictCallback):
    def __init__(self, location_generator, as_measurements=True):
        super().__init__(location_generator, OFFSET_STAGE_GLOBAL_PARAMETERS, as_measurements)


class ZDCOffsetSettingsGenerator(ValuesToSettingsDictCallback):
    # mixed offsets when using Z-drift-controller (ZDC) -> use stage coords instead of piezo
    # TODO: check if this is still the correct path, esp. z
    offset_settings_paths = OFFSET_STAGE_GLOBAL_PARAMETERS[:1] + OFFSET_SCAN_PARAMETERS[1:]

    def __init__(self, location_generator, as_measurements=True):
        super().__init__(location_generator, self.offset_settings_paths, as_measurements)


class MultipleScanOffsetsSettingsGenerator(ValuesToSettingsDictCallback):
    def __init__(self, location_generator, as_measurements=True):
        super().__init__(location_generator, OFFSET_SCAN_PARAMETERS, as_measurements, nested_generator_callback=True)


class ScanFieldSettingsGenerator(ValuesToSettingsDictCallback):

    def __init__(self, location_generator, as_measurements=True):
        super().__init__(location_generator, (OFFSET_SCAN_PARAMETERS, FOV_LENGTH_PARAMETERS), as_measurements)


if __name__ == '__main__':

    # dummy callback returning list of 3D coordinates
    positions = [[1,2,3], [4,5,6]]
    position_callback = lambda: positions

    # test ScanOffsetsSettingsGenerator / Stage... / ZDC...
    gen = ZDCOffsetSettingsGenerator(position_callback, False)
    res = gen()
    pprint.pprint(res)

    # dummy callback returning pairs of 3D coordinates
    # -> can be interpreted as offset, size for ScanFieldSettingsGenerator
    # or as pairs of offsets from MultipleScanOffsetsSettingsGenerator
    coord_pairs = [((1, 2, 3), (1, 2, 3)), ((2, 3, 4), (5, 6, 7))]
    coord_pairs_callback = lambda: coord_pairs

    # test field/ROI settings generator
    gen = ScanFieldSettingsGenerator(coord_pairs_callback, True)
    res = gen()
    pprint.pprint(res)

    # test multiple offsets generator
    gen = MultipleScanOffsetsSettingsGenerator(coord_pairs_callback, True)
    res = gen()
    pprint.pprint(res)