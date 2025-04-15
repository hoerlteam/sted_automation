import logging

from autosted.utils.tiling import relative_spiral_generator
from autosted.utils.parameter_constants import OFFSET_STAGE_GLOBAL_PARAMETERS
from autosted.callback_buildingblocks.coordinate_value_wrappers import (
    ValuesToSettingsDictCallback,
)


class SpiralOffsetGenerator:

    def __init__(
        self,
        move_size,
        start_position,
        z_position=None,
        return_parameter_dict=True,
        offset_parameter_keys=OFFSET_STAGE_GLOBAL_PARAMETERS,
    ):

        # if we get length-3 start coordinates, assume zyx and use only the second two
        # if no z position is given, re-use the one from start coords
        if len(start_position) == 3:
            if z_position is None:
                z_position = start_position[0]
            start_position = start_position[1:]

        self.start_position = list(start_position)
        self.move_size = move_size
        self.z_position = z_position
        self.return_parameter_dict = return_parameter_dict
        self.offset_parameter_keys = offset_parameter_keys

        self.location_generator = relative_spiral_generator(
            self.move_size, self.start_position
        )
        self.logger = logging.getLogger(__name__)

    def __call__(self):
        coordinates = [self.z_position] + next(self.location_generator)
        self.logger.info("new coordinates in spiral: " + str(coordinates))

        if self.return_parameter_dict:
            return ValuesToSettingsDictCallback(
                lambda: [coordinates], self.offset_parameter_keys
            )()
        else:
            return [coordinates]


class PositionListOffsetGenerator:

    # TODO: add possibility to reset index during acquisition?
    #  -> might be necessary to re-image same positions multiple times?

    def __init__(
        self,
        positions,
        auto_add_empty_z=True,
        return_parameter_dict=True,
        offset_parameter_keys=OFFSET_STAGE_GLOBAL_PARAMETERS,
    ):

        self.positions = positions
        self.auto_add_empty_z = auto_add_empty_z
        self.return_parameter_dict = return_parameter_dict
        self.offset_parameter_keys = offset_parameter_keys

        self.idx = 0
        self.logger = logging.getLogger(__name__)

    def get_all_locations(self):
        # return copy of all positions
        return list(self.positions)

    def __call__(self):

        # no more positions to image at
        if self.idx >= len(self.positions):
            return []

        # get next position and increment index
        coordinates = self.positions[self.idx]
        self.idx += 1

        # if stage positions are yx, add empty z so resulting values can be used with the
        # default zyx parameter sets
        if self.auto_add_empty_z and len(coordinates) < 3:
            coordinates = [None] * (3 - len(coordinates)) + coordinates

        self.logger.info("new coordinates from list: " + str(coordinates))

        if self.return_parameter_dict:
            return ValuesToSettingsDictCallback(
                lambda: [coordinates], self.offset_parameter_keys
            )()
        else:
            return [coordinates]
