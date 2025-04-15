import warnings

from autosted.utils.dict_utils import get_path_from_dict


OFFSET_STAGE_PARAMETERS = (
    "ExpControl/scan/range/coarse_z/off",
    "ExpControl/scan/range/coarse_y/off",
    "ExpControl/scan/range/coarse_x/off",
)

OFFSET_STAGE_GLOBAL_PARAMETERS = (
    "ExpControl/scan/range/coarse_z/g_off",
    "ExpControl/scan/range/coarse_y/g_off",
    "ExpControl/scan/range/coarse_x/g_off",
)

OFFSET_SCAN_PARAMETERS = (
    "ExpControl/scan/range/z/off",
    "ExpControl/scan/range/y/off",
    "ExpControl/scan/range/x/off",
)

OFFSET_SCAN_GLOBAL_PARAMETERS = (
    "ExpControl/scan/range/z/g_off",
    "ExpControl/scan/range/y/g_off",
    "ExpControl/scan/range/x/g_off",
)

LOCATION_PARAMETERS = (
    OFFSET_STAGE_PARAMETERS
    + OFFSET_STAGE_GLOBAL_PARAMETERS
    + OFFSET_SCAN_PARAMETERS
    + OFFSET_SCAN_GLOBAL_PARAMETERS
    + ("OlympusIX/stage", "OlympusIX/scanrange")
)

FOV_LENGTH_PARAMETERS = (
    "ExpControl/scan/range/z/len",
    "ExpControl/scan/range/y/len",
    "ExpControl/scan/range/x/len",
)

PIXEL_SIZE_PARAMETERS = (
    "ExpControl/scan/range/z/psz",
    "ExpControl/scan/range/y/psz",
    "ExpControl/scan/range/x/psz",
)

# directions of stage and scan coordinates relative to image pixels
# 1: larger pixel coord <-> larger world coord
# -1: larger pixel coord <-> smaller world coord
# NOTE: axes may be flipped in Imspector, Lightbox tiling seems to require flipped x-stage coords
# NOTE: default values (representing no flipped coords, except for scan/Piezo z)
# if SpecPy is present, we will try to load from configuration
DIRECTION_STAGE = (1, 1, 1)
DIRECTION_SCAN = (-1, 1, 1)


def try_load_stage_directions(hardware_parameters=None):

    if hardware_parameters is None:
        import specpy as sp
        hardware_parameters = sp.get_application().value_at("", sp.ValueTree.Hardware)

    x_inverted = get_path_from_dict(hardware_parameters, "OlympusIX/stage/invert_x", False)
    y_inverted = get_path_from_dict(hardware_parameters, "OlympusIX/stage/invert_y", False)
    if x_inverted is None or y_inverted is None:
        raise ValueError("could not load stage directions")

    return (1, -1 if y_inverted else 1, -1 if x_inverted else 1)


def try_load_scan_directions(hardware_parameters=None):

    if hardware_parameters is None:
        import specpy as sp
        hardware_parameters = sp.get_application().value_at("", sp.ValueTree.Hardware)

    x_flipped = get_path_from_dict(hardware_parameters, "ExpControl/calibration/scan/flip_x_axis", False)
    y_flipped = get_path_from_dict(hardware_parameters, "ExpControl/calibration/scan/flip_y_axis", False)
    z_flipped = get_path_from_dict(hardware_parameters, "ExpControl/calibration/scan/flip_z_axis", False)

    if x_flipped is None or y_flipped is None or z_flipped is None:
        raise ValueError("could not load scan directions")

    # NOTE: flipped z axis -> same direction as stage (1)
    return (1 if z_flipped else -1, -1 if y_flipped else 1, -1 if x_flipped else 1)


# try to load stage and scan directions
try:
    DIRECTION_STAGE = try_load_stage_directions()
except:
    warnings.warn(f"Could not load stage directions from SpecPy. Defaulting to {DIRECTION_STAGE}")

try:
    DIRECTION_SCAN = try_load_scan_directions()
except:
    warnings.warn(f"Could not load scan directions from SpecPy. Defaulting to {DIRECTION_SCAN}")
