import logging

import numpy as np
from skimage.measure import regionprops

from pipeline2.utils.parameter_constants import (PIXEL_SIZE_PARAMETERS, OFFSET_SCAN_PARAMETERS, FOV_LENGTH_PARAMETERS)
from pipeline2.callback_buildingblocks.coordinate_value_wrappers import ValuesToSettingsDictCallback
from pipeline2.detection.spot_detection import refill_ignored_dimensions, pixel_to_physical_coordinates
from pipeline2.utils.dict_utils import get_path_from_dict
from pipeline2.data import MeasurementData

from calmutils.misc import filter_rprops

class SegmentationWrapper:

    fov_length_parameter_paths = FOV_LENGTH_PARAMETERS
    pixel_size_parameter_paths = PIXEL_SIZE_PARAMETERS

    def __init__(self, data_source_callback, detection_function, configurations=(0,), channels=(0,), detection_kwargs=None,
                 regionprops_filters=None, reference_configuration=None, offset_parameter_paths=OFFSET_SCAN_PARAMETERS,
                 plot_detections=True, return_parameter_dict=True):

        self.data_source_callback = data_source_callback
        self.detection_function = detection_function
        self.offset_parameter_paths = offset_parameter_paths
        self.plot_detections = plot_detections
        self.return_parameter_dict = return_parameter_dict

        # do not invert any dimension by default
        self.invert_dimensions = (False,) * len(offset_parameter_paths)

        # make sure we have a sequence of configurations & channels, even if just a single one is selected
        self.configurations = (configurations,) if np.isscalar(configurations) else configurations
        self.channels = (channels,) if np.isscalar(channels) else channels

        # by default pick the first configuration as reference
        # if user gave a configuration index, make sure it is among those we load
        self.reference_configuration = self.configurations[0] if reference_configuration is None else reference_configuration
        if self.reference_configuration not in self.configurations:
            raise ValueError('Reference configuration must be one of: {}'.format(self.configurations))

        # keyword arguments that are passed to the detection_function call (after the images)
        self.detection_kwargs = {} if detection_kwargs is None else detection_kwargs

        # dict of property name -> (min, max) ranges for regionprops, default to empty dict
        self.regionprops_filters = {} if regionprops_filters is None else regionprops_filters

        self.logger = logging.getLogger(__name__)

    def run_segmentation(self, images):

        label_map = self.detection_function(*images, **self.detection_kwargs)

        bboxes = []
        for region in regionprops(label_map):
            if filter_rprops(region, self.regionprops_filters):
                bboxes.append(region.bbox)

        if self.plot_detections:
            # TODO: plot
            pass

        return bboxes

    def __call__(self):

        # get list of images we want to process
        data = self.data_source_callback()
        images = MeasurementData.collect_images_from_measurement_data(data, self.configurations, self.channels)

        # check for singleton dimensions
        # (dropped in squeezed images, but should be refilled with original position in results)
        singleton_dims = MeasurementData.get_singleton_dimensions(data, self.reference_configuration)
        self.logger.debug(f'Singleton dimensions: {singleton_dims}')

        # get settings dict for reference configuration
        measurement_settings = data.measurement_settings[self.reference_configuration]

        # run segmentation, should return sequence of (min_0, min_1, ..., max_0, max_1, ...) bounding boxes
        bboxes = self.run_segmentation(images)
        self.logger.info(f'detected {len(bboxes)} bounding box(es), pixel coordinates: {bboxes}')

        # we found nothing, return
        if len(bboxes) == 0:
            return []

        # get reference frame in world units
        offsets = np.array([get_path_from_dict(measurement_settings, path, False) for path in self.offset_parameter_paths], dtype=float)
        fov_lengths = np.array([get_path_from_dict(measurement_settings, path, False) for path in self.fov_length_parameter_paths], dtype=float)
        pixel_sizes = np.array([get_path_from_dict(measurement_settings, path, False) for path in self.pixel_size_parameter_paths], dtype=float)

        self.logger.debug(f'reference offsets: {offsets}')
        self.logger.debug(f'reference fov lengths: {fov_lengths}')
        self.logger.debug(f'reference pixel sizes: {pixel_sizes}')

        results = []

        for bbox in bboxes:

            # split bbox into min/max
            mins, maxs = bbox[:len(bbox)//2], bbox[len(bbox)//2:]

            # refill ignored/singleton dimensions, convert to world units
            mins, maxs = refill_ignored_dimensions(mins, singleton_dims), refill_ignored_dimensions(maxs, singleton_dims)
            mins = pixel_to_physical_coordinates(mins, offsets, fov_lengths, pixel_sizes, ignore_dimension=singleton_dims, invert_dimension=self.invert_dimensions)
            maxs = pixel_to_physical_coordinates(maxs, offsets, fov_lengths, pixel_sizes, ignore_dimension=singleton_dims, invert_dimension=self.invert_dimensions)

            # min/max to center/length
            roi_center = (mins + maxs) / 2
            roi_length = np.abs(maxs - mins)

            results.append((roi_center, roi_length))

        self.logger.info(f'physical unit center/length of detection(s): {results}')

        if self.return_parameter_dict:
            return ValuesToSettingsDictCallback(lambda: results, (self.offset_parameter_paths, self.fov_length_parameter_paths))()
        else:
            return results


class ROIDetectorWrapper(SegmentationWrapper):

    def run_segmentation(self, images):
        bboxes = self.detection_function(*images, **self.detection_kwargs)
        return bboxes


if __name__ == '__main__':

    from pprint import pprint
    import logging
    logging.basicConfig(level=logging.DEBUG)

    img = np.zeros((1, 1, 201, 201), dtype=float)

    img[0, 0, 100:101, 100:101] = 2
    # img[0, 0, 20:25, 50:55] = 10

    off = [0, 0, 0]
    pixel_size = [0.1, 0.1, 0.1]
    fov = np.array([0.1, 0.1, 0.1]) * (201 - 1)

    settings_call = ValuesToSettingsDictCallback(lambda: ((off, pixel_size, fov),),
                                                 (OFFSET_SCAN_PARAMETERS, PIXEL_SIZE_PARAMETERS, FOV_LENGTH_PARAMETERS))
    measurement_settings, hardware_settings = settings_call()[0][0]

    data = MeasurementData()
    data.append(hardware_settings, measurement_settings, [img])
    data_call = lambda: data

    def detection_fun(img, thresh=0):
        from scipy.ndimage import label
        return label(img > thresh)[0]

    def detection_rois(img, thresh=0):
        from scipy.ndimage import label
        return [r.bbox for r in regionprops(label(img > thresh)[0])]

    callback = SegmentationWrapper(data_call, detection_fun, return_parameter_dict=True, detection_kwargs={'thresh': 0.1},
                                   regionprops_filters={'area': (24, 199)})
    callback = ROIDetectorWrapper(data_call, detection_rois, return_parameter_dict=True,
                                   detection_kwargs={'thresh': 1})
    callback.invert_dimensions = (False, True, True)


    res = callback()

    pprint(res)
