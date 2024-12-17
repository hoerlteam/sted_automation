from pipeline2.utils.coordinate_utils import virtual_bbox_from_settings
from calmutils.misc.bounding_boxes import get_iou

class AlreadyImagedFOVFilter:

    def __init__(self, pipeline, level, iou_thresh, z_ignore=False):
        self.iou_thresh = iou_thresh
        self.pipeline = pipeline
        self.lvl = level
        self.z_ignore = z_ignore

        # init old BBox list
        self.old_bboxes = []

    def check(self, task):

        self.update_old_bboxes()

        # get BBoxes of new task
        bboxes_new = []

        for i in range(len(task)):
            measurement_updates, _ = task[i]
            (min_i, len_i) = virtual_bbox_from_settings(measurement_updates)
            if self.z_ignore:
                    min_i = min_i[1:]
                    len_i = len_i[1:]
            bboxes_new.append((min_i, len_i))

        # if any overlap > IOU threshold: return False, else True
        for bbox in bboxes_new:
            for bbox_old in self.old_bboxes:
                #print('check against old BBOX IOU: {}'.format(self.get_iou(bbox, bbox_old, self.ignore_z)))
                if get_iou(bbox, bbox_old) > self.iou_thresh:
                    return False
        return True

    def update_old_bboxes(self):
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
                (min_i, len_i) = virtual_bbox_from_settings(setts_i)
                if self.z_ignore:
                    min_i = min_i[1:]
                    len_i = len_i[1:]
                self.old_bboxes.append((min_i, len_i))
