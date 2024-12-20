import numpy as np

from autosted.utils.dict_utils import get_parameter_value_array_from_dict
from autosted.utils.parameter_constants import FOV_LENGTH_PARAMETERS, OFFSET_SCAN_GLOBAL_PARAMETERS, OFFSET_STAGE_GLOBAL_PARAMETERS, OFFSET_SCAN_PARAMETERS, DIRECTION_SCAN, DIRECTION_STAGE, OFFSET_STAGE_PARAMETERS, PIXEL_SIZE_PARAMETERS


def pixel_to_physical_coordinates(pixel_coordinates, offset, pixel_size, fov_size=None, shape=None, ignore_dimension=None, invert_dimension=None):
    """
    Transform pixel coordinates in an image to physical coordinates usable as Imspector parameters.
    Pixel coordinates will be added to physical offset of an image which in Imspector corresponds to the center of the image.

    :param pixel_coordinates: pixel coordinates (array-like)
    :param offset: Imspector metadata offset (array-like)
    :param pixel_size: Imspector metadata pixel-size (array-like)
    :param fov_size: Imspector metadata FOV-length (array-like)
    :param shape: pixel shape of the image (array-like)
    :param ignore_dimension: dimensions to ignore (keep offset) (boolean array-like)
    :param invert_dimension: dimensions to invert (boolean array-like)
    :return: x in world coordinates (array-like)
    """

    # defaults for fov size / shape: we need one
    if fov_size is None and shape is None:
        raise ValueError('Either physical FOV size or pixel shape have to be provided')
    # alternative: only shape is given -> multiply with pixel size to get fov
    if fov_size is None:
        fov_size = np.array(list(shape)) * np.array(pixel_size)

    # default for ignore_dimension: don't ignore any dimension
    if ignore_dimension is None:
        ignore_dimension = np.zeros_like(pixel_coordinates, dtype=bool)
    # default for invert_dimension: don't invert any dimension
    if invert_dimension is None:
        invert_dimension = np.zeros_like(pixel_coordinates, dtype=bool)

    # make everything array
    pixel_coordinates = np.array(pixel_coordinates, dtype=float)
    offset = np.array(offset, dtype=float)
    fov_size = np.array(fov_size, dtype=float)
    pixel_size = np.array(pixel_size, dtype=float)
    ignore_dimension = np.array(ignore_dimension, dtype=bool)
    invert_dimension = np.array(invert_dimension, dtype=bool)

    return (offset  # old offset
            - np.logical_not(ignore_dimension) * np.where(invert_dimension, -1, 1) * fov_size / 2.0  # minus half length
            + np.logical_not(ignore_dimension) * np.where(invert_dimension, -1, 1) * pixel_coordinates * pixel_size) # new offset in units


def get_offset_parameters_defaults(offset_parameters='scan'):

    # use stage or scan parameter paths if specified via string
    if offset_parameters == 'scan':
        offset_parameter_paths = OFFSET_SCAN_PARAMETERS
    elif offset_parameters == 'stage':
        offset_parameter_paths = OFFSET_STAGE_GLOBAL_PARAMETERS
    else:
        offset_parameter_paths = offset_parameters # use as-is if not preset

    # TODO: raise exception on wrong input (not sequence of strings)?

    # boolean tuple of inverted dimensions
    # use default stage / scan directions
    if offset_parameters == 'scan':
        invert_dimensions = tuple(direction < 0 for direction in DIRECTION_SCAN)
    elif offset_parameters == 'stage':
        invert_dimensions = tuple(direction < 0 for direction in DIRECTION_STAGE)
    else:
        # if no preset is selected do not invert any dimension by default
        invert_dimensions = (False,) * len(offset_parameters)

    return offset_parameter_paths, invert_dimensions


def refill_ignored_dimensions(coordinates, ignored_dimensions, fill_value=0.0):
    """
    Given a vector coordinates and a (longer) boolean vector,
    add fill_value to the coordinates at positions where ignored_dimesnions is True
    """
    coords_refilled = []
    true_coords_idx = 0
    for is_ignored_dim in ignored_dimensions:
        if is_ignored_dim:
            coords_refilled.append(fill_value)
        else:
            coords_refilled.append(coordinates[true_coords_idx])
            true_coords_idx += 1
    return coords_refilled


def virtual_bbox_from_settings(settings):
    """
    Get a minimum, FOV length bounding box from Imspector settings
    NOTE: we flip offests of axes that do not move in same direction as image pixel coordinates
    that way, two bounding boxes can be checked for overlap, but the virtual origin does not correspond to the real location
    """

    # direction tuples to arrays
    direction_scan, direction_stage = np.array(list(DIRECTION_SCAN), dtype=float), np.array(list(DIRECTION_STAGE), dtype=float)

    offs_stage = get_parameter_value_array_from_dict(settings, OFFSET_STAGE_PARAMETERS)
    offs_stage_global = get_parameter_value_array_from_dict(settings, OFFSET_STAGE_GLOBAL_PARAMETERS)
    offs_stage_total = offs_stage + offs_stage_global

    offs_scan = get_parameter_value_array_from_dict(settings, OFFSET_SCAN_PARAMETERS)
    offs_scan_global = get_parameter_value_array_from_dict(settings, OFFSET_SCAN_GLOBAL_PARAMETERS)
    offs_scan_total = offs_scan + offs_scan_global

    fov_len = get_parameter_value_array_from_dict(settings, FOV_LENGTH_PARAMETERS)

    offset_combined = offs_stage_total * direction_stage + offs_scan_total * direction_scan
    start = offset_combined - fov_len / 2

    return start, fov_len


def approximate_pixel_shift_from_settings(settings_reference, settings_moving):
    """
    Get the approximate pixel offset of image with Imspector settings settings_moving from reference image with settings_reference
    """

    start_moving, _ = virtual_bbox_from_settings(settings_moving)
    start_reference, _ = virtual_bbox_from_settings(settings_reference)

    pixel_sizes = get_parameter_value_array_from_dict(settings_reference, PIXEL_SIZE_PARAMETERS)
    pixel_off = ((start_moving - start_reference) / pixel_sizes).astype(int)

    return pixel_off