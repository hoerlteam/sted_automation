import logging

import numpy as np

from pipeline2.callback_buildingblocks.coordinate_value_wrappers import ValuesToSettingsDictCallback
from pipeline2.detection.display_util import draw_detections_multicolor, draw_detections_1c, DEFAULT_COLOR_NAMES
from pipeline2.utils.dict_utils import get_path_from_dict
from pipeline2.data import MeasurementData
from pipeline2.utils.parameter_constants import (PIXEL_SIZE_PARAMETERS, OFFSET_SCAN_PARAMETERS, FOV_LENGTH_PARAMETERS)


class CoordinateDetectorWrapper:

    fov_length_parameter_paths = FOV_LENGTH_PARAMETERS
    pixel_size_parameter_paths = PIXEL_SIZE_PARAMETERS

    def __init__(self, data_source_callback, detection_function, configurations=(0,), channels=(0,), detection_kwargs=None,
                 reference_configuration=None, offset_parameters=OFFSET_SCAN_PARAMETERS, plot_detections=True):
        """
        Parameters
        ----------
        data_source_callback : a callable (e.g. NewestDataSelector), which should return a MeasurementData object
        detection_function : a callable taking one or more images as the first positional arguments and optionally keyword arguments
        configurations : index of configuration to use, or list of indices to use multiple
        channels : index of channel to use, or list of indices to use multiple
        detection_kwargs : keyword arguments to pass to detection_function
        reference_configuration : index of configuration from which to get metadata (e.g. stage/scan position)
        offset_parameters : parameter paths of offset position in measurement settings
        """

        self.data_source_callback = data_source_callback
        self.detection_function = detection_function

        self.offset_parameter_paths = offset_parameters
        # do not invert any dimension by default
        self.invert_dimensions = (False,) * len(offset_parameters)

        # keyword arguments that are passed to the detection_function call (after the images)
        self.detection_kwargs = {} if detection_kwargs is None else detection_kwargs

        # make sure we have a sequence of configurations & channels, even if just a single one is selected
        self.configurations = (configurations,) if np.isscalar(configurations) else configurations
        self.channels = (channels,) if np.isscalar(channels) else channels

        # by default pick the first configuration as reference
        # if user gave a configuration index, make sure it is among those we load
        self.reference_configuration = self.configurations[0] if reference_configuration is None else reference_configuration
        if self.reference_configuration not in self.configurations:
            raise ValueError('Reference configuration must be one of: {}'.format(self.configurations))

        self.plot_detections = plot_detections

        self.normalization_range = (0.5, 99.5)
        self.plot_colors = DEFAULT_COLOR_NAMES

    def to_world_coordinates(self, detections_pixel, setts, ignore_dim):

        offsets = np.array([get_path_from_dict(setts, path, False) for path in self.offset_parameter_paths], dtype=float)
        fov_lengths = np.array([get_path_from_dict(setts, path, False) for path in self.fov_length_parameter_paths], dtype=float)
        pixel_sizes = np.array([get_path_from_dict(setts, path, False) for path in self.pixel_size_parameter_paths], dtype=float)

        res = []
        for detection in detections_pixel:
            detection = np.array(detection, dtype=float)
            res.append(pixel_to_physical_coordinates(detection, offsets, fov_lengths, pixel_sizes, ignore_dim, self.invert_dimensions))
        return res

    def __call__(self):

        # get list of images we want to process
        data = self.data_source_callback()
        images = MeasurementData.collect_images_from_measurement_data(data, self.configurations, self.channels)

        # check for singleton dimensions
        # (dropped in squeezed images, but should be refilled with original position in results)
        singleton_dims = MeasurementData.get_singleton_dimensions(data, self.reference_configuration)

        # get settings dict for reference configuration
        measurement_settings = data.measurement_settings[self.reference_configuration]

        # run detection
        detections = self.detection_function(*images, **self.detection_kwargs)

        if self.plot_detections:
            # get images of reference configuration
            imgs_ref_config = MeasurementData.collect_images_from_measurement_data(data, (self.reference_configuration,), self.channels)
            # plot in RGB (multiple channels) or gray (single channel)
            if len(imgs_ref_config) > 1:
                draw_detections_multicolor(imgs_ref_config, detections, self.normalization_range, None, 3, True, self.plot_colors)
            else:
                draw_detections_1c(imgs_ref_config[0], detections, self.normalization_range, None, 3, True)

        # nothing found -> early return
        if len(detections) == 0:
            return []

        # we support nested detections, i.e. a list of lists of coordinates
        # check by looking at the first element of the first detection
        # in non-nested results, this will be a number, otherwise it will be an array of coords
        nested_detections = not np.isscalar(detections[0][0])

        # go over pixel results, convert to world units
        # if we have nested results, do an extra inner loop to keep structure
        if not nested_detections:
            detections = [refill_ignored_dimensions(detection_pixel, singleton_dims) for detection_pixel in detections]
            detections_unit = self.to_world_coordinates(detections, measurement_settings, singleton_dims)
            return detections_unit
        else:
            results = []
            for detections_i in detections:
                detections_i = [refill_ignored_dimensions(detections_i_pixel, singleton_dims) for detections_i_pixel in detections_i]
                detection_unit = self.to_world_coordinates(detections_i, measurement_settings, singleton_dims)
                results.append(detection_unit)
            return results


def refill_ignored_dimensions(coordinates, ignore_dim):
    coords_refilled = []
    true_coords_i = 0
    for d_ignored in ignore_dim:
        if d_ignored:
            coords_refilled.append(0.0)
        else:
            coords_refilled.append(coordinates[true_coords_i])
            true_coords_i += 1
    return coords_refilled


def pixel_to_physical_coordinates(pixel_coordinates, offset, fov_size, pixel_size, ignore_dimension=None, invert_dimension=None):
    """
    correct pixel coordinates to physical coordinates usable as Imspector parameters

    :param pixel_coordinates: pixel coordinates (array-like)
    :param offset: Imspector metadata offset (array-like)
    :param fov_size: Imspector metadata FOV-length (array-like)
    :param pixel_size: Imspector metadata pixel-size (array-like)
    :param ignore_dimension: dimensions to ignore (keep offset) (boolean array-like)
    :param invert_dimension: dimensions to invert (boolean array-like)
    :return: x in world coordinates (array-like)
    """

    pixel_coordinates = np.array(pixel_coordinates, dtype=float)
    offset = np.array(offset, dtype=float)
    fov_size = np.array(fov_size, dtype=float)
    pixel_size = np.array(pixel_size, dtype=float)

    # default for ignore_dimension: don't ignore any dimension
    if ignore_dimension is None:
        ignore_dimension = np.zeros_like(pixel_coordinates, dtype=bool)
    # default for invert_dimension: don't invert any dimension
    if invert_dimension is None:
        invert_dimension = np.zeros_like(pixel_coordinates, dtype=bool)

    ignore_dimension = np.array(ignore_dimension, dtype=bool)
    invert_dimension = np.array(invert_dimension, dtype=bool)

    return (offset  # old offset
            - np.logical_not(ignore_dimension) * np.where(invert_dimension, -1, 1) * fov_size / 2.0  # minus half length
            + np.logical_not(ignore_dimension) * np.where(invert_dimension, -1, 1) * pixel_coordinates * pixel_size) # new offset in units


if __name__ == '__main__':

    from pipeline2.data import MeasurementData
    from pprint import pprint

    logging.basicConfig(level=logging.DEBUG)

    img = np.zeros((1, 1, 201, 201), dtype=float)
    img[0, 0, 100, 100] = 5
    img[0, 0, 20, 50] = 5

    img_ch2 = img.copy()

    off = [0, 0, 0]
    pixel_size = [0.1, 0.1, 0.1]
    fov = np.array([0.1, 0.1, 0.1]) * 200
    settings_call = ValuesToSettingsDictCallback(lambda: ((off, pixel_size, fov),),
                                                 (OFFSET_SCAN_PARAMETERS, PIXEL_SIZE_PARAMETERS, FOV_LENGTH_PARAMETERS))
    measurement_settings, hardware_settings = settings_call()[0][0]


    data = MeasurementData()
    data.append(hardware_settings, measurement_settings, [img, img_ch2])
    data_call = lambda: data

    def fun(img, *other_imgs, sigma=3):
        from scipy.ndimage import gaussian_laplace
        from skimage.feature import peak_local_max

        for oi in other_imgs:
            print(oi.shape)

        return peak_local_max(-gaussian_laplace(img.astype(float), sigma), threshold_abs=1e-6)

    detector = CoordinateDetectorWrapper(data_call, fun, channels=(0,1), detection_kwargs={'sigma': 3})
    #
    # detector = LegacySpotPairFinder(data_call, 1, [500, 0.1], plot_detections=True, return_parameter_dict=True)
    detector.normalization_range = (0, 100)
    # detector.plot_colors = ('cyan', 'magenta')

    res = detector()
    # res = ParameterValuesRepeater(SimpleManualOffset(detector, [13,13,13]), 2, nested=False)()
    pprint(res)
