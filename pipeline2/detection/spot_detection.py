import logging

import numpy as np
from matplotlib import pyplot as plt
from scipy import ndimage as ndi

from pipeline2.taskgeneration.coordinate_building_blocks import ValuesToSettingsDictCallback
from pipeline2.detection.spot_util import detect_blobs_find_pairs, detect_blobs
from pipeline2.detection.display_util import draw_detections_multicolor, draw_detections_1c, DEFAULT_COLOR_NAMES
from pipeline2.utils.dict_utils import get_path_from_dict
from pipeline2.utils.parameter_constants import (PIXEL_SIZE_PARAMETERS, OFFSET_STAGE_GLOBAL_PARAMETERS,
                                                 OFFSET_SCAN_PARAMETERS, FOV_LENGTH_PARAMETERS)


class SimpleFocusPlaneDetector:

    def __init__(self, data_source_callback, configuration=0, channel=0, invert_z_direction=True):
        """


        Parameters
        ----------
        data_source_callback : an object implementing get_data(), which should return a MeasurementData object
        configuration : int, index of configuration to use for focus
        channel : int, index of channel to use for focus
        verbose : bool, set to True for extra debugging output
        invert_z_direction: whether to invert the z direction of the focus update
            if the update should change stage focus, but the image stack was acquired using the piezo drive this should be True

        """
        self.data_source_callback = data_source_callback
        self.configuration = configuration
        self.channel = channel
        self.logger = logging.getLogger(__name__)
        self.invert_z_direction = invert_z_direction

        self.offset_z_path = OFFSET_STAGE_GLOBAL_PARAMETERS[0]
        self.pixel_size_z_path = PIXEL_SIZE_PARAMETERS[0]

    @staticmethod
    def mean_along_axis(img, axis):
        """
        calculate mean of every hyper-slice in image along axis
        """
        axes = tuple([ax for ax in range(len(img.shape)) if ax != axis])
        profile = np.mean(img, axes)
        return profile

    @staticmethod
    def focus_in_stack(img, pixel_size, axis=0, sigma=3):
        # get mean profile, smooth it via a Gaussian blur
        profile = SimpleFocusPlaneDetector.mean_along_axis(img, axis)
        smooth_profile = ndi.gaussian_filter1d(profile, sigma=sigma, mode='constant')
        profile_max = np.argmax(smooth_profile)
        # calculate offset of maximum in comparison to middle
        pix_d = profile_max - ((len(profile) - 1) / 2)
        return pix_d * pixel_size

    def __call__(self):

        data = self.data_source_callback()

        # no data yet -> empty update
        if data is None:
            self.logger.info(': No data for Z correction present -> skipping.')
            return [[None, None, None]]

        if (data.num_configurations <= self.configuration) or (data.num_channels(self.configuration) <= self.channel):
            raise ValueError('no images present. TODO: fail gracefully/skip here')

        # get image of selected configuration and channel and convert to float
        img = data.data[self.configuration][self.channel][0, :, :, :]
        img = np.array(img, float)

        # 2D image -> empty update
        if img.shape[0] <= 1:
            self.logger.info(': Image is 2D, cannot do Z correction -> skipping.')
            return [[None, None, None]]

        # get old z-offset and pixel size
        setts = data.measurementSettings[self.configuration]
        z_offset_old = get_path_from_dict(setts, self.offset_z_path, keep_structure=False)
        z_pixel_size = get_path_from_dict(setts, self.pixel_size_z_path, keep_structure=False)

        # get z delta, add or subtract from old z-offset
        z_delta = self.focus_in_stack(img, z_pixel_size, 0)
        new_z = z_offset_old + z_delta * (-1 if self.invert_z_direction else 1)

        self.logger.info(': Corrected Focus (was {}, new {})'.format(z_offset_old, new_z))
        
        return [[new_z, None, None]]


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

    @staticmethod
    def collect_images_from_measurement_data(data, configurations, channels):
        images = []
        for configuration in configurations:
            if configuration >= data.num_configurations:
                raise ValueError('Requested configuration does not exist in MeasurementData')
            for channel in channels:
                if channel >= data.num_channels(configuration):
                    raise ValueError('Requested channel does not exist in MeasurementData')
                img = data.data[configuration][channel]
                img = img.squeeze()
                images.append(img)
        return images

    def __call__(self):

        # get list of images we want to process
        data = self.data_source_callback()
        images = CoordinateDetectorWrapper.collect_images_from_measurement_data(data, self.configurations, self.channels)

        # check if any dimensions of first image of reference channel are singleton
        # NOTE: we ignore the first of the 4 dimensions of the Imspector stack
        singleton_dims = np.array(data.data[self.reference_configuration][0].shape) == 1
        singleton_dims = singleton_dims[1:]

        # get settings dict for reference configuration
        measurement_settings = data.measurement_settings[self.reference_configuration]

        # run detection
        detections = self.detection_function(*images, **self.detection_kwargs)

        if self.plot_detections:
            # get images of reference configuration
            imgs_ref_config = CoordinateDetectorWrapper.collect_images_from_measurement_data(data, (self.reference_configuration,), self.channels)
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


class SimpleSingleChannelSpotDetector:
    """
    simple Laplacian-of-Gaussian spot detector with an additional intensity criterion (spots must be x-fold brighter than median around spot)
    will detect spots in one channel, if more channels are specified, detection will be applied independently and merged.
    """
    offset_parameter_paths = OFFSET_SCAN_PARAMETERS
    pixel_size_parameter_paths = PIXEL_SIZE_PARAMETERS
    invert_dimensions = (False, False, False)

    def __init__(self, data_source_callback, sigmas, threshold, configuration=0, channel=0,
                 median_threshold=3, median_radius=5, refine_detections=True, return_parameter_dict=False,
                 plot_detections=False):

        self.data_source_callback = data_source_callback
        self.sigmas = sigmas
        self.thresholds = threshold
        self.configuration = configuration
        self.channels = channel
        self.median_threshold = median_threshold
        self.median_radius = median_radius
        self.plot_detections = plot_detections
        self.logger = logging.getLogger(__name__)
        self.refine_detections = refine_detections
        self.return_parameter_dict = return_parameter_dict

        # ensure we have a list of channels / thresholds
        # NOTE: this way, we can do single channel detection in multiple channels independently
        if np.isscalar(self.channels):
            self.channels = [self.channels]
        if np.isscalar(self.thresholds):
            self.thresholds = [self.thresholds]
    
    def do_plot(self, spots_pixel, img):

        draw_detections_1c(img, spots_pixel, None, None, 3)
        plt.show()

    def to_world_coordinates(self, localizations, settings, ignore_dim, img_shape):

        offset_image = np.array([get_path_from_dict(settings, off_path, keep_structure=False) for off_path in self.offset_parameter_paths], dtype=float)
        psz_image = np.array([get_path_from_dict(settings, off_path, keep_structure=False) for off_path in self.pixel_size_parameter_paths], dtype=float)
        fov_length = (np.array(img_shape, dtype=float) - 1) * psz_image

        self.logger.debug('offset: {}'.format(offset_image))
        self.logger.debug('pixel size: {}'.format(psz_image))
        self.logger.debug('FOV length: {}'.format(fov_length))
            
        res = []
        for loc in localizations:
            loc = np.array(loc, dtype=float)
            res.append(pixel_to_physical_coordinates(loc, offset_image, fov_length, psz_image, ignore_dim, self.invert_dimensions))
        return res

    def __call__(self):

        data = self.data_source_callback()
        if (data.num_configurations <= self.configuration) or (data.num_channels(self.configuration) < max(self.channels)):
            raise ValueError('required images not present. TODO: fail gracefully/skip here')

        locs = []
        for channel, threshold in zip(self.channels, self.thresholds):

            # get image for channel, make float
            img = data.data[self.configuration][channel][0, :, :, :]
            img = np.array(img, float)
            # get measurement setting for channel
            setts = data.measurement_settings[self.configuration]
            # check which dimensions are singleton
            ignore_dim = np.array([d == 1 for d in img.shape])
            
            # if sigma is scalar: repeat for number of 'valid' dimensions
            sigmas = self.sigmas
            if np.isscalar(sigmas):
                sigmas = [sigmas] * int(len(ignore_dim) - np.sum(ignore_dim))
            
            # discard singleton dimensions for detection
            img_squeezed = np.squeeze(img)

            # do detection
            locs_per_channel = detect_blobs(img_squeezed, sigmas, threshold, False, self.median_threshold,
                                            self.median_radius, self.refine_detections)

            # re-introduce zeroes to get back to 3-d (if we dropped dims)
            locs_expanded = []
            for loc in locs_per_channel:
                loc_expanded = refill_ignored_dimensions(loc, ignore_dim)
                locs_expanded.append(loc_expanded)
            locs_per_channel = locs_expanded

            # accumulate localizations for all channels
            locs += locs_per_channel

        # to physical coordinates
        corrected = self.to_world_coordinates(locs, setts, ignore_dim, img.shape) if len(locs) > 1 else []

        self.logger.info(self.__class__.__name__ + ': found {} spots. pixel coordinates:'.format(len(locs)))
        for loc in locs:
            self.logger.info(loc)

        self.logger.info(self.__class__.__name__ + ': found {} spots. physical coordinates:'.format(len(locs)))
        for locC in corrected:
            self.logger.info(locC)

        # plot
        # NOTE: this only shows the last channel if we detect in multiple
        # TODO: plot better
        if self.plot_detections:
            self.do_plot(locs, img)

        if not self.return_parameter_dict:
            return corrected
        else:
            return ValuesToSettingsDictCallback(lambda: corrected, self.offset_parameter_paths)()


class LegacySpotPairFinder:
    """
    wrapper for the 'old' spot pair detector
    get_locations will return a list of coordinate lists
    of scan coordinates (stage coordinates are ignored)
    """

    offset_parameter_paths = OFFSET_SCAN_PARAMETERS
    pixel_size_parameter_paths = PIXEL_SIZE_PARAMETERS
    fov_length_parameter_paths = FOV_LENGTH_PARAMETERS
    invert_dimensions = (False, False, False)

    def __init__(self, data_source, sigma, thresholds, median_thresholds=(3, 3), median_radius=5, channels=(0, 1),
                 in_channel_min_distance=3, between_channel_max_distance=5, plot_detections=False, return_parameter_dict=False):

        self.data_source_callback = data_source
        self.sigma = sigma
        self.thresholds = thresholds
        self.median_thresholds = median_thresholds
        self.median_radius = median_radius
        self.plot_detections = plot_detections
        self.channels = channels
        self.between_channel_max_distance = between_channel_max_distance
        self.in_channel_min_distance = in_channel_min_distance
        self.return_parameter_dict = return_parameter_dict

        self.normalization_range = (0.5, 99.5)
        self.plot_colors = ('red', 'green')

        self.logger = logging.getLogger(__name__)

    def do_plot(self, locations_pixel, stack1, stack2):
        draw_detections_multicolor([stack1, stack2], locations_pixel, self.normalization_range, None, 3,
                                   percentile_range=True, color_names=self.plot_colors)

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
        data = self.data_source_callback()
        if (data.num_configurations < 1) or (data.num_channels(0) < 2):
            raise ValueError(
                'too few images for LegacySpotPairFinder. The MeasurementData provided needs to have two images in the first configuration.')
        stack1 = data.data[0][self.channels[0]][0, :, :, :]
        stack2 = data.data[0][self.channels[1]][0, :, :, :]

        # make float
        stack1 = np.array(stack1, float)
        stack2 = np.array(stack2, float)

        setts = data.measurement_settings[0]

        pairsRaw = detect_blobs_find_pairs(stack1, stack2, self.sigma, self.thresholds, False, False, self.median_thresholds,
                                           self.median_radius,
                                           in_channel_min_distance=self.in_channel_min_distance,
                                           between_channel_max_distance=self.between_channel_max_distance)

        self.logger.info(': found {} spot pairs. pixel coordinates:'.format(len(pairsRaw)))
        for pr in pairsRaw:
            self.logger.info(pr)

        # plot
        if self.plot_detections:
            self.do_plot(pairsRaw, stack1, stack2)

        ignore_dim = np.array([d for d in stack1.shape]) == 1
        corrected = self.to_world_coordinates(pairsRaw, setts, ignore_dim)

        self.logger.info(self.__class__.__name__ + ': found {} spot pairs. offsets:'.format(len(pairsRaw)))
        for pc in corrected:
            self.logger.info(pc)

        if not self.return_parameter_dict:
            return corrected
        else:
            return ValuesToSettingsDictCallback(lambda: corrected, self.offset_parameter_paths)()


class PairedLegacySpotPairFinder(LegacySpotPairFinder):

    def __call__(self):
        data = self.data_source_callback()
        if (data.num_configurations < 1) or (data.num_channels(0) < 2):
            raise ValueError(
                'too few images for LegacySpotPairFinder. The MeasurementData provided needs to have two images in the first configuration.')
        stack1 = data.data[0][self.channels[0]][0, :, :, :]
        stack2 = data.data[0][self.channels[1]][0, :, :, :]

        # make float
        stack1 = np.array(stack1, float)
        stack2 = np.array(stack2, float)

        setts = data.measurement_settings[0]

        pairsRaw = detect_blobs_find_pairs(stack1, stack2, self.sigma, self.thresholds, False, False, self.median_thresholds,
                                           self.median_radius, True,
                                           in_channel_min_distance=self.in_channel_min_distance,
                                           between_channel_max_distance=self.between_channel_max_distance)

        self.logger.info(self.__class__.__name__ + ': found {} spot pairs. pixel coordinates:'.format(len(pairsRaw)))
        for pr in pairsRaw:
            self.logger.info(pr)

        # plot
        if self.plot_detections:
            self.do_plot([list((np.array(p[0]) + np.array(p[1])) / 2) for p in pairsRaw], stack1, stack2)

        ignore_dim = np.array([d for d in stack1.shape]) == 1
        corrected_1 = self.to_world_coordinates([p[0] for p in pairsRaw], setts, ignore_dim)
        corrected_2 = self.to_world_coordinates([p[1] for p in pairsRaw], setts, ignore_dim)

        corrected = list(zip(corrected_1, corrected_2))

        self.logger.info(self.__class__.__name__ + ': found {} spot pairs. offsets:'.format(len(pairsRaw)))
        for pc in corrected:
            self.logger.info(pc)

        if not self.return_parameter_dict:
            return corrected
        else:
            return ValuesToSettingsDictCallback(lambda: corrected, self.offset_parameter_paths, nested_generator_callback=True)()


class ZDCSpotPairFinder(LegacySpotPairFinder):

    # TODO: check if the coordinates / directions used in ZDC mode are still correct
    offset_parameter_paths = OFFSET_STAGE_GLOBAL_PARAMETERS[:1] + OFFSET_SCAN_PARAMETERS[1:]
    invert_dimensions = (True, False, False)


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

    # default for ignore_dimension: don't ignore any dimension
    if ignore_dimension is None:
        ignore_dimension = np.zeros_like(pixel_coordinates, dtype=bool)
    # default for invert_dimension: don't invert any dimension
    if invert_dimension is None:
        invert_dimension = np.zeros_like(pixel_coordinates, dtype=bool)

    pixel_coordinates = np.array(pixel_coordinates, dtype=float)
    offset = np.array(offset, dtype=float)
    fov_size = np.array(fov_size, dtype=float)
    pixel_size = np.array(pixel_size, dtype=float)

    ignore_dimension = np.array(ignore_dimension, dtype=bool)
    invert_dimension = np.array(invert_dimension, dtype=bool)

    return (offset  # old offset
            - np.logical_not(ignore_dimension) * np.where(invert_dimension, -1, 1) * fov_size / 2.0  # minus half length
            + np.logical_not(ignore_dimension) * np.where(invert_dimension, -1, 1) * pixel_coordinates * pixel_size) # new offset in units


if __name__ == '__main__':

    from pipeline2.data import MeasurementData
    from pprint import pprint

    logging.basicConfig(level=logging.ERROR)
    logging.getLogger(__name__).setLevel(logging.DEBUG)
    logging.getLogger('pipeline2').setLevel(logging.DEBUG)

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

    detector = SimpleSingleChannelSpotDetector(data_call, 1, 0.1, plot_detections=True, return_parameter_dict=False)
    detector.invert_dimensions = (False, False, True)

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
