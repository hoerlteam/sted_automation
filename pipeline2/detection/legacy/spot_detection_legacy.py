import logging

import numpy as np
from matplotlib import pyplot as plt

from pipeline2.callback_buildingblocks.coordinate_value_wrappers import ValuesToSettingsDictCallback
from pipeline2.detection.display_util import draw_detections_1c, draw_detections_multicolor
from pipeline2.detection.spot_detection import pixel_to_physical_coordinates, refill_ignored_dimensions
from pipeline2.detection.spot_util import detect_blobs, detect_blobs_find_pairs
from pipeline2.utils.dict_utils import get_path_from_dict
from pipeline2.utils.parameter_constants import OFFSET_SCAN_PARAMETERS, PIXEL_SIZE_PARAMETERS, FOV_LENGTH_PARAMETERS, \
    OFFSET_STAGE_GLOBAL_PARAMETERS


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

        # get measurement setting for configuration
        settings = data.measurement_settings[self.configuration]

        locs = []
        for channel, threshold in zip(self.channels, self.thresholds):

            # get image for channel, make float
            img = data.data[self.configuration][channel][0, :, :, :]
            img = np.array(img, float)

            # check which dimensions are singleton
            ignore_dim = np.array([d == 1 for d in img.shape])

            # if sigma is scalar: repeat for number of 'valid' dimensions
            sigmas = self.sigmas
            if np.isscalar(sigmas):
                sigmas = [sigmas] * int(len(ignore_dim) - np.sum(ignore_dim))

            # discard singleton dimensions for detection
            img_squeezed = np.squeeze(img)

            # do detection
            locs_for_channel = detect_blobs(img_squeezed, sigmas, threshold, normalize=False, threshold_rel_median=self.median_threshold,
                                            med_radius=self.median_radius, refine=self.refine_detections)

            # re-introduce zeroes to get back to 3-d (if we dropped dims)
            locs_expanded = []
            for loc in locs_for_channel:
                loc_expanded = refill_ignored_dimensions(loc, ignore_dim)
                locs_expanded.append(loc_expanded)
            locs_for_channel = locs_expanded

            # accumulate localizations for all channels
            locs += locs_for_channel

        # to physical coordinates
        corrected = self.to_world_coordinates(locs, settings, ignore_dim, img.shape) if len(locs) > 1 else []

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


if __name__ == '__main__':

    from pipeline2.data import MeasurementData
    from pprint import pprint

    logging.basicConfig(level=logging.INFO)

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
    #
    # detector = LegacySpotPairFinder(data_call, 1, [500, 0.1], plot_detections=True, return_parameter_dict=True)
    detector.normalization_range = (0, 100)
    # detector.plot_colors = ('cyan', 'magenta')

    res = detector()
    # res = ParameterValuesRepeater(SimpleManualOffset(detector, [13,13,13]), 2, nested=False)()
    pprint(res)