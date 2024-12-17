import numpy as np
from calmutils.stitching import stitch
from calmutils.stitching.fusion import fuse_image
from calmutils.stitching.transform_helpers import translation_matrix
from calmutils.stitching.phase_correlation import get_axes_aligned_bbox
from calmutils.misc.bounding_boxes import get_overlap_bounding_box

from pipeline2.callback_buildingblocks.data_selection import NewestDataSelector
from pipeline2.data import MeasurementData
from pipeline2.utils.dict_utils import merge_dicts, generate_nested_dict, get_parameter_value_array_from_dict
from pipeline2.utils.coordinate_utils import get_offset_parameters_defaults
from pipeline2.utils.parameter_constants import (OFFSET_SCAN_PARAMETERS, OFFSET_SCAN_GLOBAL_PARAMETERS,
                                                 OFFSET_STAGE_PARAMETERS, OFFSET_STAGE_GLOBAL_PARAMETERS,
                                                 FOV_LENGTH_PARAMETERS, PIXEL_SIZE_PARAMETERS,
                                                 DIRECTION_SCAN, DIRECTION_STAGE)


class StitchedNewestDataSelector(NewestDataSelector):
    """
    Callback that will select the newest MeasurementData of a given level and
    return virtually stitched data with all neighboring images of the same level 
    """

    def __init__(self, pipeline, level, channel=0, configuration=0, offset_parameters='scan', register_tiles=True):
        super().__init__(pipeline, level)
        self.channel = channel
        self.configuration = configuration
        
        self.offset_parameter_paths, inverted_dimensions = get_offset_parameters_defaults(offset_parameters)
        self.offset_directions = np.where(inverted_dimensions, -1, 1)

        self.register_tiles = register_tiles
        # background value for areas with no images in fusion
        # by setting it to an "unnatural number" like -1, we can distinguish areas that are not imaged yet
        self.background_value = -1

    def __call__(self):

        # get the newest data, return None if not present
        data_newest = super().__call__()
        if data_newest is None:
            return None

        # virtual bbox of reference
        settings_reference = data_newest.measurement_settings[self.configuration]
        (min_r, len_r) = virtual_bbox_from_settings(settings_reference)

        # get all other indices of same level
        index_length = self.pipeline.hierarchy_levels.index(self.level) + 1
        indices_same_level = [k for k in self.pipeline.data.keys() if len(k) == index_length]

        # get all overlapping data
        data_other = []
        for idx in indices_same_level:
            data_other_i = self.pipeline.data.get(idx, None)

            # virtual bbox of other image
            setts_i = data_other_i.measurement_settings[self.configuration]
            (min_i, len_i) = virtual_bbox_from_settings(setts_i)

            # check overlap
            overlap = (get_overlap_bounding_box(len_r, len_i, min_r, min_i)) is not None
            if overlap:
                data_other.append(data_other_i)

        # if no overlapping views, just return original data
        if len(data_other) < 1:
            return data_newest

        # get pixel offsets and images to stitch
        imgs_other = []
        offs_other = []
        for data_other_i in data_other:
            setts_i = data_other_i.measurement_settings[self.configuration]
            off_i = approximate_pixel_shift_from_settings(settings_reference, setts_i)
            img_i = np.squeeze(data_other_i.data[self.configuration][self.channel])

            imgs_other.append(img_i)
            offs_other.append(off_i)

        # get reference image
        reference_img = np.squeeze(data_newest.data[self.configuration][self.channel])

        imgs = [reference_img] + imgs_other
        pixel_off_reference = [0] * len(reference_img.shape)
        pixel_offsets = [pixel_off_reference] + offs_other
        pixel_offsets = [np.array(offset) for offset in pixel_offsets]

        # get transformations
        if self.register_tiles:
            transforms = stitch(imgs, pixel_offsets)
        else:
            transforms = [translation_matrix(offset) for offset in pixel_offsets]

        # stitch / fuse
        stitched_mins, stitched_maxs = get_axes_aligned_bbox([img.shape for img in imgs], transforms)
        stitched_mins = np.floor(stitched_mins).astype(int)
        stitched_maxs = np.ceil(stitched_maxs).astype(int)
        
        bbox = list(zip(stitched_mins, stitched_maxs))
        stitched = fuse_image(imgs, transforms, bbox=bbox, oob_val=self.background_value)

        min_rev = np.array(stitched_mins, dtype=float)
        len_orig_half = np.array(reference_img.shape, dtype=float)/2
        len_stitched_half = np.array(stitched.shape, dtype=float)/2

        # additional pixel offset of center of stitched image relative to center of reference image
        additional_off = len_stitched_half - (len_orig_half - min_rev)

        # to world units
        pixel_sizes = get_parameter_value_array_from_dict(settings_reference, PIXEL_SIZE_PARAMETERS)

        additional_off *= pixel_sizes
        additional_off *= self.offset_directions

        new_len = len_stitched_half * 2 * pixel_sizes

        # use stage or scan offsets as basis for dummy offsets
        offs_to_use = get_parameter_value_array_from_dict(settings_reference, self.offset_parameter_paths)

        # create dummy settings
        stitch_setts = merge_dicts(settings_reference)
        for i, (p_off, p_len) in enumerate(zip(self.offset_parameter_paths, FOV_LENGTH_PARAMETERS)):
            stitch_setts = merge_dicts(stitch_setts, generate_nested_dict(new_len[i], p_off))
            stitch_setts = merge_dicts(stitch_setts, generate_nested_dict(offs_to_use[i] + additional_off[i], p_len))

        # stitched image to Imspector shape
        # add singleton (T) dimension
        res_img = stitched.reshape((1, ) + stitched.shape)
        # add None for images of other channels
        res_data = [None] * self.channel + [res_img] + [None] * (data_newest.num_channels(self.configuration) - (self.channel + 1))

        # wrap results, use None for other configs
        res = MeasurementData()
        for _ in range(self.configuration):
            res.append(None, None, None)
        res.append(data_newest.hardware_settings[self.configuration], stitch_setts, res_data)
        for _ in range(data_newest.num_configurations -(self.configuration + 1)):
            res.append(None, None, None)

        return res


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
