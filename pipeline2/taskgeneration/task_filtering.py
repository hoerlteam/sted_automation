import numpy as np

from ..utils.dict_utils import merge_dicts
from ..callback_buildingblocks.stitched_data_selection import _virtual_bbox_from_settings, get_overlap_bounding_box


class AlreadyImagedFOVFilter:

    def __init__(self, pipeline, level, iou_thresh, z_ignore=False):
        self.iou_thresh = iou_thresh
        self.pipeline = pipeline
        self.lvl = level
        self.z_ignore = z_ignore

        # init old BBox list
        self.old_bboxes = []

    def check(self, task):

        # get BBoxes of new task
        bboxes_new = []
        updates = task.get_all_updates()
        for update in updates:
            measUpdates, _ = update
            measUpdates = merge_dicts(*measUpdates)
            (min_i, len_i) = _virtual_bbox_from_settings(measUpdates)
            bboxes_new.append((min_i, len_i))

        # if any overlap > IOU threshold: return False, else True
        for bbox in bboxes_new:
            for bbox_old in self.old_bboxes:
                #print('check against old BBOX IOU: {}'.format(self.get_iou(bbox, bbox_old, self.ignore_z)))
                if self.get_iou(bbox, bbox_old, self.z_ignore) > self.iou_thresh:
                    return False
        return True

    def update(self):
        """
        get BBoxes for all existing acquisitions
        """
        # rest list
        self.old_bboxes.clear()

        # get all other indices of same level
        len_of_idx = self.pipeline.hierarchy_levels.levels.index(self.lvl) + 1
        idxes_same_level = [idx for idx in self.pipeline.data.keys() if len(idx) == len_of_idx]
        for idx in idxes_same_level:
            data_other_i = self.pipeline.data.get(idx, None)
            # go through all confs
            for setts_i in data_other_i.measurementSettings:
                # virtual bbox of image
                (min_i, len_i) = _virtual_bbox_from_settings(setts_i)
                self.old_bboxes.append((min_i, len_i))

    @staticmethod
    def get_iou(bbox1, bbox2, z_ignore=False):
        (min1, len1) = bbox1
        (min2, len2) = bbox2

        #print(bbox1)
        #print(bbox2)
            
        overlap = get_overlap_bounding_box(len1, len2, min1, min2)

        # no overlap
        if overlap is None:
            return 0

        r_min, r_max = overlap
        
        # if user decided to ignore z, use only x & y
        if z_ignore:
            r_min = r_min[:-1]
            r_max = r_max[:-1]
            len1 = len1[:-1]
            len2 = len2[:-1]
            
        len_ol = np.array(r_max, dtype=float) - np.array(r_min, dtype=float)
        area_o = np.prod(len_ol)
        area_u = np.prod(len1) + np.prod(len2) - area_o

        return area_o / area_u
