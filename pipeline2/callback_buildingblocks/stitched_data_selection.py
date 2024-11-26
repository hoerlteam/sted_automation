import numpy as np
from calmutils.stitching import stitch
from calmutils.stitching.fusion import fuse_image
from calmutils.stitching.transform_helpers import translation_matrix
from calmutils.stitching.phase_correlation import get_axes_aligned_bbox

from pipeline2.callback_buildingblocks.data_selection import NewestDataSelector
from pipeline2.data import MeasurementData
from pipeline2.utils.dict_utils import merge_dicts, get_path_from_dict, generate_nested_dict
from pipeline2.utils.parameter_constants import (OFFSET_SCAN_PARAMETERS, OFFSET_SCAN_GLOBAL_PARAMETERS,
                                                 OFFSET_STAGE_PARAMETERS, OFFSET_STAGE_GLOBAL_PARAMETERS,
                                                 FOV_LENGTH_PARAMETERS, PIXEL_SIZE_PARAMETERS)


STAGE_DIRECTIONS = np.array([-1, 1, 1], dtype=float)


class StitchedNewestDataSelector(NewestDataSelector):
    """
    Callback that will select the newest MeasurementData of a given level and
    return virtually stitched data with all neighboring images of the same level 
    """

    def __init__(self, pipeline, level, channel=0, configuration=0, generate_stage_offsets=False, register_tiles=True):
        super().__init__(pipeline, level)
        self.channel = channel
        self.configuration = configuration
        self.generate_stage_offsets = generate_stage_offsets
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
        (min_r, len_r) = _virtual_bbox_from_settings(settings_reference)

        # get all other indices of same level
        index_length = self.pipeline.hierarchy_levels.index(self.level) + 1
        indices_same_level = [k for k in self.pipeline.data.keys() if len(k) == index_length]

        # get all overlapping data
        data_other = []
        for idx in indices_same_level:
            data_other_i = self.pipeline.data.get(idx, None)

            # virtual bbox of other image
            setts_i = data_other_i.measurementSettings[self.configuration]
            (min_i, len_i) = _virtual_bbox_from_settings(setts_i)

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
            off_i = _approx_offset_from_settings(settings_reference, setts_i)
            img_i = np.squeeze(data_other_i.data[self.configuration][self.channel])

            imgs_other.append(img_i)
            offs_other.append(off_i)

        # get reference image
        reference_img = np.squeeze(data_newest.data[self.configuration][self.channel])

        imgs = [reference_img] + imgs_other
        pixel_off_reference = [0] * len(reference_img.shape)
        pixel_offsets = [pixel_off_reference] + offs_other

        # get transformations
        if self.register_tiles:
            transforms = stitch(imgs, pixel_offsets)
        else:
            transforms = [translation_matrix(offset) for offset in pixel_offsets]

        # stitch / fuse
        stitched_mins, stitched_maxs = get_axes_aligned_bbox([img.shape for img in imgs], transforms)
        bbox = list(zip(stitched_mins, stitched_maxs))
        stitched = fuse_image(bbox, imgs, transforms, oob_val=self.background_value)

        min_rev = np.array(stitched_mins, dtype=float)
        len_orig_half = np.array(reference_img.shape, dtype=float)/2
        len_stitched_half = np.array(stitched.shape, dtype=float)/2

        # additional pixel offset of center of stitched image relative to center of reference image
        additional_off = len_stitched_half - (len_orig_half - min_rev)

        # to world units
        offs_stage_global = np.array([get_path_from_dict(settings_reference, path, False) for path in OFFSET_STAGE_GLOBAL_PARAMETERS],
                                     dtype=float)
        offs_scan = np.array([get_path_from_dict(settings_reference, path, False) for path in OFFSET_SCAN_PARAMETERS],
                             dtype=float)
        pixel_sizes = np.array([get_path_from_dict(settings_reference, path, False) for path in PIXEL_SIZE_PARAMETERS],
                               dtype=float)

        additional_off *= pixel_sizes
        if self.generate_stage_offsets:
            additional_off *= STAGE_DIRECTIONS

        new_len = len_stitched_half * 2 * pixel_sizes

        # use stage or scan offsets as basis for dummy offsets
        offs_to_use = offs_stage_global if self.generate_stage_offsets else offs_scan
        offset_paths_to_use = OFFSET_STAGE_GLOBAL_PARAMETERS if self.generate_stage_offsets else OFFSET_SCAN_PARAMETERS

        # create dummy settings
        stitch_setts = merge_dicts(settings_reference)
        for i, (p_off, p_len) in enumerate(zip(offset_paths_to_use, FOV_LENGTH_PARAMETERS)):
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


def _virtual_bbox_from_settings(settings):
    """
    Get a minimum and FOV length from Imspector settings
    NB: since we move `down` in stack, we calculate a virtual origin here
        that way, two bounding boxes can be checked for overlap, but the virtual origin does not correspond to the real location
    """
    offs_stage = np.array([get_path_from_dict(settings, path, False) for path in OFFSET_STAGE_PARAMETERS], dtype=float)
    offs_stage_global = np.array([get_path_from_dict(settings, path, False) for path in OFFSET_STAGE_GLOBAL_PARAMETERS], dtype=float)

    offs_scan = np.array([get_path_from_dict(settings, path, False) for path in OFFSET_SCAN_PARAMETERS], dtype=float)
    offs_scan_global = np.array([get_path_from_dict(settings, path, False) for path in OFFSET_SCAN_GLOBAL_PARAMETERS], dtype=float)

    fov_len = np.array([get_path_from_dict(settings, path, False) for path in FOV_LENGTH_PARAMETERS], dtype=float)

    offs = (offs_stage + offs_stage_global) * STAGE_DIRECTIONS + offs_scan + offs_scan_global
    start = offs - fov_len / 2

    return start, fov_len


def _approx_offset_from_settings(setts_ref, setts2):
    """
    Get the approximate pixel offset of image with Imspector settings setts2 from reference image with setts_ref
    """

    start_i, _ = _virtual_bbox_from_settings(setts2)
    start_r, _ = _virtual_bbox_from_settings(setts_ref)

    pixel_sizes = np.array([get_path_from_dict(setts_ref, path, False) for path in PIXEL_SIZE_PARAMETERS], dtype=float)
    pixel_off = ((start_i - start_r) / pixel_sizes).astype(int)

    return pixel_off


def get_overlap_bounding_box(length_1, length_2, offset_1=None, offset_2=None):

    # if no offsets are given, assume all zero
    if offset_1 is None:
        offset_1 = [0] * len(length_1)
    if offset_2 is None:
        offset_2 = [0] * len(length_2)

    res_min = []
    res_max = []

    for d in range(len(length_1)):

        min_1 = offset_1[d]
        min_2 = offset_2[d]
        max_1 = min_1 + length_1[d]
        max_2 = min_2 + length_2[d]

        min_ol = max(min_1, min_2)
        max_ol = min(max_1, max_2)

        # no overlap in any one dimension -> return None
        if max_ol < min_ol:
            return None

        res_min.append(min_ol)
        res_max.append(max_ol)

    return res_min, res_max
