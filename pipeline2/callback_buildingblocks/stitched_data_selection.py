import numpy as np

from calmutils.stitching import stitch
from calmutils.stitching import stitching

from .data_selection import NewestDataSelector
from ..data import MeasurementData
from ..utils.dict_utils import update_dicts, get_path_from_dict, generate_recursive_dict

STAGE_DIRECTIONS = np.array([1,1,-1], dtype=float)

class StitchedNewestDataSelector(NewestDataSelector):
    """
    Callback that will select the newest MeasurementData of a given level and
    return virtually stitched data with all neighboring images of the same level 
    """

    def __init__(self, pipeline, level, channel=0, configuration=0, generate_stage_offsets=False):
        super().__init__(pipeline, level)
        self.channel = channel
        self.configuration = configuration
        self.generate_stage_offsets = generate_stage_offsets

    def __call__(self):

        # get newest data, return None if not present
        data_newest: MeasurementData = super().__call__()
        if data_newest is None:
            return None

        # virtual bbox of reference
        setts = data_newest.measurement_settings[self.configuration]
        (min_r, len_r) = _virtual_bbox_from_settings(setts)

        # get all other indices of same level
        index_length = self.pipeline.hierarchy_levels.index(self.level) + 1
        indices_same_level = [k for k in self.pipeline.data.keys() if len(k) == index_length]
        
        #print('Virtual BBOX ref: {}, {}'.format(min_r, len_r))
        # get all overlapping data
        data_other = []
        for idx in indices_same_level:
            data_other_i = self.pipeline.data.get(idx, None)

            # virtual bbox of image
            setts_i = data_other_i.measurementSettings[self.configuration]
            (min_i, len_i) = _virtual_bbox_from_settings(setts_i)
            #print('Virtual BBOX test: {}, {}'.format(min_i, len_i))

            # check overlap
            overlap = (_get_overlaps(len_r, len_i, min_r, min_i)) is not None

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
            off_i = list(reversed(list(_approx_offset_from_settings(setts, setts_i))))
            img_i = np.squeeze(data_other_i.data[self.configuration][self.channel])

            imgs_other.append(img_i)
            offs_other.append(off_i)

        img = np.squeeze(data_newest.data[self.configuration][self.channel])
        off = [0] * len(img.shape)

        # stitch
        stitched, shift, min, corrs = stitch(img, imgs_other, off, offs_other)
        
        print('corrs: {}'.format(corrs))

        min_rev = np.array(list(reversed(min)), dtype=float)
        len_orig_half = np.array(list(reversed(img.shape)), dtype=float)/2
        len_stitched_half = np.array(list(reversed(stitched.shape)), dtype=float)/2

        additional_off = len_stitched_half - (len_orig_half - min_rev)

        # to pixel units
        offs_scan = np.array([get_path_from_dict(
            setts, 'ExpControl/scan/range/{}/off'.format(c), False) for c in ['x', 'y', 'z']], dtype=float)
        offs_stage = np.array([get_path_from_dict(
            setts, 'ExpControl/scan/range/coarse_{}/g_off'.format(c), False) for c in ['x', 'y', 'z']], dtype=float)
        pixel_sizes = np.array([get_path_from_dict(
            setts, 'ExpControl/scan/range/{}/psz'.format(c), False) for c in ['x', 'y', 'z']], dtype=float)

        additional_off *= pixel_sizes
        new_len = len_stitched_half * 2 * pixel_sizes

        # use stage or scan offsets as basis for dummy offsets
        offs_to_use = offs_stage if self.generate_stage_offsets else offs_scan

        # create dummy settings
        stitch_setts = update_dicts(setts)
        for i, d in enumerate(['x', 'y', 'z']):
            stitch_setts = update_dicts(stitch_setts, generate_recursive_dict(new_len[i], 'ExpControl/scan/range/{}/len'.format(d)))
            stitch_setts = update_dicts(stitch_setts, generate_recursive_dict(offs_to_use[i] + additional_off[i], 'ExpControl/scan/range/{}/off'.format(d)))

        # add singleton (T) dimension
        res_img = stitched.reshape((1, ) + stitched.shape)
        # add None for images of other channels
        res_data = [None] * self.channel + [res_img] + [None] * (data_newest.num_images(self.configuration) - (self.channel + 1))

        # wrap results, use None for other configs
        res = MeasurementData()
        for _ in range(self.configuration):
            res.append(None, None, None)
        res.append(data_newest.hardware_settings[self.configuration], stitch_setts, res_data)
        for _ in range(data_newest.num_configurations -(self.configuration + 1)):
            res.append(None, None, None)

        return res


def _virtual_bbox_from_settings(setts):
    """
    Get a minimum and FOV length from Imspector settings
    NB: since we move `down` in stack, we calculate a virtual origin here
        that way, two bounding boxes can be checked for overlap, bot the virtual origin does not correspond to the real location
    """
    offs_stage = np.array([get_path_from_dict(
        setts, 'ExpControl/scan/range/coarse_{}/g_off'.format(c), False) for c in ['x', 'y', 'z']],
        dtype=float)
    offs_scan = np.array([get_path_from_dict(
        setts, 'ExpControl/scan/range/{}/off'.format(c), False) for c in ['x', 'y', 'z']], dtype=float)
    offs_global = np.array([get_path_from_dict(
        setts, 'ExpControl/scan/range/{}/g_off'.format(c), False) for c in ['x', 'y', 'z']], dtype=float)
    fov_len = np.array([get_path_from_dict(
        setts, 'ExpControl/scan/range/{}/len'.format(c), False) for c in ['x', 'y', 'z']], dtype=float)
    pixel_sizes = np.array([get_path_from_dict(
        setts, 'ExpControl/scan/range/{}/psz'.format(c), False) for c in ['x', 'y', 'z']], dtype=float)

    offs = offs_stage * STAGE_DIRECTIONS + offs_global + offs_scan
    start = offs - fov_len / 2

    return start, fov_len


def _approx_offset_from_settings(setts_ref, setts2):
    """
    Get the approximate pixel offset of image with Imspector settings setts2 from reference image with setts_ref
    """
    offs_stage_r = np.array([get_path_from_dict(
        setts_ref, 'ExpControl/scan/range/coarse_{}/g_off'.format(c), False) for c in ['x', 'y', 'z']],
        dtype=float)
    offs_stage_i = np.array([get_path_from_dict(
        setts2, 'ExpControl/scan/range/coarse_{}/g_off'.format(c), False) for c in ['x', 'y', 'z']],
        dtype=float)
    offs_stage = (offs_stage_i - offs_stage_r) * STAGE_DIRECTIONS

    offs_scan_r = np.array([get_path_from_dict(
        setts_ref, 'ExpControl/scan/range/{}/off'.format(c), False) for c in ['x', 'y', 'z']], dtype=float)
    offs_scan_i = np.array([get_path_from_dict(
        setts2, 'ExpControl/scan/range/{}/off'.format(c), False) for c in ['x', 'y', 'z']], dtype=float)
    offs_scan = offs_scan_i - offs_scan_r


    offs_global_r = np.array([get_path_from_dict(
        setts_ref, 'ExpControl/scan/range/{}/g_off'.format(c), False) for c in ['x', 'y', 'z']], dtype=float)
    offs_global_i = np.array([get_path_from_dict(
        setts2, 'ExpControl/scan/range/{}/g_off'.format(c), False) for c in ['x', 'y', 'z']], dtype=float)
    offs_global = offs_global_i - offs_global_r

    pixel_sizes = np.array([get_path_from_dict(
        setts_ref, 'ExpControl/scan/range/{}/psz'.format(c), False) for c in ['x', 'y', 'z']], dtype=float)

    pixel_off = ((offs_scan + offs_stage + offs_global) / pixel_sizes).astype(int)

    return pixel_off

def _get_overlaps(len1, len2, off1=None, off2=None):
    if off1 is None:
        off_1 = [0] * len(len1)

    if off2 is None:
        off_2 = [0] * len(len2)

    r_min = []
    r_max = []

    for d in range(len(len1)):
        min_1 = off1[d]
        min_2 = off2[d]
        max_1 = min_1 + len1[d]
        max_2 = min_2 + len2[d]

        min_ol = max(min_1, min_2)
        max_ol = min(max_1, max_2)

        if max_ol < min_ol:
            return None

        r_min.append(min_ol)
        r_max.append(max_ol)

    return r_min, r_max