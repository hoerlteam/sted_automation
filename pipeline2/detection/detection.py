import numpy as np
import collections
from itertools import cycle
from matplotlib import pyplot as plt
from scipy import ndimage as ndi

from pipeline2.taskgeneration.coordinate_building_blocks import  ValuesToSettingsDictCallback
from pipeline2.detection.spot_util import pair_finder_inner, detect_blobs
from pipeline2.detection.display_util import draw_detections_2c, draw_detections_1c
from pipeline2.utils.dict_utils import get_path_from_dict
from pipeline2.utils.parameter_constants import PIXEL_SIZE_PARAMETERS, OFFSET_STAGE_GLOBAL_PARAMETERS, OFFSET_SCAN_PARAMETERS


class ParameterValuesRepeater:
    """
    Simple callback to wrap another callback and repeat the returned values n times.

    E.g., can be used to image each location returned by a spot or ROI detector multiple times.
    """
    
    def __init__(self, wrapped_callback, n=2, nested=False):
        self.wrapped_callback = wrapped_callback
        self.n = n
        self.nested = nested
        
    def __call__(self):
        value_sets = self.wrapped_callback()
        repeated_values = []
        
        for values in value_sets:

            # case 1: add multiple copies of values wrapped in a tuple,
            # this way, the nested values can become configurations in a wrapping building block
            if self.nested:
                repeated_values.append((values,) * self.n)
            # case 2 (default): just add multiple copies, that is intended to result in multiple configurations
            else:
                for _ in range(self.n):
                    repeated_values.append(values)
        
        return repeated_values
    
    
class SimpleManualOffset:

    """
    Callback to wrap another callback and add a manual offset to the returned parameter values.

    E.g., can be used to add a focus offset [z_offset, 0, 0] to results of spot detection
    """
    
    def __init__(self, wrapped_callback, offset):
        self.wrapped_callback = wrapped_callback
        self.offset = np.array(offset, dtype=float)
        
    def __call__(self):
        values = self.wrapped_callback()
        res = []
        
        offset_cycler = cycle(self.offset)
        for values_i in values:

            # we have a sequence of collection types (e.g., list of lists of coordinates)
            # add offset to all, keep structure
            if len(values_i) > 0 and not np.isscalar(values_i[0]):
                ri_tup = []
                for values_i_inner in values_i:
                    ri = []
                    for value in values_i_inner:
                        off_i = next(offset_cycler)
                        ri.append(None if value is None else value + off_i)
                    ri_tup.append(ri)
                res.append(tuple(ri_tup))

            # we have just a list of value sequences (e.g., coordinate lists/arrays)
            else:
                ri = []
                for value in values_i:
                    off_i = next(offset_cycler)
                    ri.append(None if value is None else value + off_i)
                res.append(ri)
        return res


class SimpleFocusPlaneDetector:

    def __init__(self, data_source_callback, configuration=0, channel=0, verbose=False, invert_z_direction=True):
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
        self.verbose = verbose
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

        data = self.data_source_callback.get_data()

        # no data yet -> empty update
        if data is None:
            if self.verbose:
                print(self.__class__.__name__ + ': No data for Z correction present -> skipping.')
            return [[None, None, None]]

        if (data.num_configurations <= self.configuration) or (data.num_images(self.configuration) <= self.channel):
            raise ValueError('no images present. TODO: fail gracefully/skip here')

        # get image of selected configuration and channel and convert to float
        img = data.data[self.configuration][self.channel][0, :, :, :]
        img = np.array(img, np.float)

        # 2D image -> empty update
        if img.shape[0] <= 1:
            if self.verbose:
                print(self.__class__.__name__ + ': Image is 2D, cannot do Z correction -> skipping.')
            return [[None, None, None]]

        # get old z-offset and pixel size
        setts = data.measurementSettings[self.configuration]
        z_offset_old = get_path_from_dict(setts, self.offset_z_path, keep_structure=False)
        z_pixel_size = get_path_from_dict(setts, self.pixel_size_z_path, keep_structure=False)

        # get z delta, add or subtract from old z-offset
        z_delta = self.focus_in_stack(img, z_pixel_size, 0)
        new_z = z_offset_old + z_delta * (-1 if self.invert_z_direction else 1)

        if self.verbose:
            print(self.__class__.__name__ + ': Corrected Focus (was {}, new {})'.format(z_offset_old, new_z))
        
        return [[new_z, None, None]]


class CoordinateDetectorWrapper:
    def __init__(self, data_source_callback, detection_function_callback):
        self.data_source_callback = data_source_callback
        self.detection_function_callback = detection_function_callback



class ROIDetectorWrapper:
    pass


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
                 plot_detections=False, verbose=False):

        self.data_source_callback = data_source_callback
        self.sigmas = sigmas
        self.thresholds = threshold
        self.configuration = configuration
        self.channels = channel
        self.median_threshold = median_threshold
        self.median_radius = median_radius
        self.plot_detections = plot_detections
        self.verbose = verbose
        self.refine_detections = refine_detections
        self.return_parameter_dict = return_parameter_dict

        # ensure we have a list of channels / thresholds
        # NOTE: this way, we can do single channel detection in multiple channels independently
        if np.isscalar(self.channels):
            self.channels = [self.channels]
        if np.isscalar(self.thresholds):
            self.thresholds = [self.thresholds]
    
    def do_plot(self, spots_pixel, img):

        draw_detections_1c(img, spots_pixel, None, 0, 3)
        plt.show()

    def to_world_coordinates(self, localizations, settings, ignore_dim, img_shape):

        offset_image = np.array([get_path_from_dict(settings, off_path, keep_structure=False) for off_path in self.offset_parameter_paths], dtype=float)
        psz_image = np.array([get_path_from_dict(settings, off_path, keep_structure=False) for off_path in self.pixel_size_parameter_paths], dtype=float)
        fov_length = (np.array(img_shape, dtype=float) - 1) * psz_image

        if self.verbose:
            print('offset: {}'.format(offset_image))
            print('pixel size: {}'.format(psz_image))
            print('FOV length: {}'.format(fov_length))
            
        res = []
        for loc in localizations:
            loc = np.array(loc, dtype=float)
            res.append(pixel_to_physical_coordinates(loc, offset_image, fov_length, psz_image, ignore_dim, self.invert_dimensions))
        return res

    def __call__(self):

        data = self.data_source_callback()
        if (data.num_configurations <= self.configuration) or (data.num_images(self.configuration) < max(self.channels)):
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
        corrected = self.to_world_coordinates(locs, setts, ignore_dim, img.shape)

        if self.verbose:
            print(self.__class__.__name__ + ': found {} spots. pixel coordinates:'.format(len(locs)))
            for loc in locs:
                print(loc)

        if self.verbose:
            print(self.__class__.__name__ + ': found {} spots. physical coordinates:'.format(len(locs)))
            for locC in corrected:
                print(locC)

        # plot
        # NOTE: this only shows the last channel if we detect in multiple
        # TODO: plot better
        if self.plot_detections:
            self.do_plot(locs, img)

        if not self.return_parameter_dict:
            return corrected
        else:
            return ValuesToSettingsDictCallback(lambda: corrected, self.offset_parameter_paths)()


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


class LegacySpotPairFinder():
    """
    wrapper for the 'old' spot pair detector
    get_locations will return a list of coordinate lists
    of scan coordinates (stage coordinates are ignored)
    """

    def __init__(self, dataSource, sigma, thresholds, medianThresholds=[3, 3], medianRadius=5, channels=(0,1),
                 in_channel_min_distance=3, between_channel_max_distance=5):
        self.dataSource = dataSource
        self.sigma = sigma
        self.thresholds = thresholds
        self.medianThresholds = medianThresholds
        self.medianRadius = medianRadius
        self.plotDetections = False
        self.verbose = False
        self.channels = channels
        self.between_channel_max_distance = between_channel_max_distance
        self.in_channel_min_distance = in_channel_min_distance


    def doPlot(self, pairsPixel, stack1, stack2):
        draw_detections_2c(stack1, stack2, [s[-1::-1] for s in pairsPixel], [0.5, 99.99], 0, 3, percentile_range=True)

    def correctForOffset(self, pairsPixel, setts, ignore_dim):
        offsOld = np.array([get_path_from_dict(
            setts, 'ExpControl/scan/range/{}/off'.format(c), False) for c in ['x', 'y', 'z']], dtype=float)

        lensOld = np.array([get_path_from_dict(
            setts, 'ExpControl/scan/range/{}/len'.format(c), False) for c in ['x', 'y', 'z']], dtype=float)

        pszOld = np.array([get_path_from_dict(
            setts, 'ExpControl/scan/range/{}/psz'.format(c), False) for c in ['x', 'y', 'z']], dtype=float)

        res = []
        for pair in pairsPixel:
            pairT = np.array(pair, dtype=float)
            res.append(pixel_to_physical_coordinates(pairT, offsOld, lensOld, pszOld, ignore_dim))
        return res

    def get_locations(self):
        data = self.dataSource.get_data()
        if (data.num_configurations < 1) or (data.num_images(0) < 2):
            raise ValueError(
                'too few images for LegacySpotPairFinder. The RichData provided needs to have two images in the first configuration.')
        stack1 = data.data[0][self.channels[0]][0, :, :, :]
        stack2 = data.data[0][self.channels[1]][0, :, :, :]

        # make float
        stack1 = np.array(stack1, np.float)
        stack2 = np.array(stack2, np.float)

        setts = data.measurementSettings[0]

        pairsRaw = pair_finder_inner(stack1, stack2, self.sigma, self.thresholds, True, False, self.medianThresholds,
                                     self.medianRadius,
                                     in_channel_min_distance=self.in_channel_min_distance,
                                     between_channel_max_distance=self.between_channel_max_distance)

        if self.verbose:
            print(self.__class__.__name__ + ': found {} spot pairs. pixel coordinates:'.format(len(pairsRaw)))
            for pr in pairsRaw:
                print(pr)

        # plot
        if self.plotDetections:
            self.doPlot(pairsRaw, stack1, stack2)

        ignore_dim = np.array([d for d in stack1.shape][-1::-1]) == 1
        corrected = self.correctForOffset(pairsRaw, setts, ignore_dim)

        if self.verbose:
            print(self.__class__.__name__ + ': found {} spot pairs. offsets:'.format(len(pairsRaw)))
            for pc in corrected:
                print(pc)

        return corrected


class PairedLegacySpotPairFinder(LegacySpotPairFinder):

    def get_locations(self):
        data = self.dataSource.get_data()
        if (data.num_configurations < 1) or (data.num_images(0) < 2):
            raise ValueError(
                'too few images for LegacySpotPairFinder. The RichData provided needs to have two images in the first configuration.')
        stack1 = data.data[0][self.channels[0]][0, :, :, :]
        stack2 = data.data[0][self.channels[1]][0, :, :, :]

        # make float
        stack1 = np.array(stack1, np.float)
        stack2 = np.array(stack2, np.float)

        setts = data.measurementSettings[0]

        pairsRaw = pair_finder_inner(stack1, stack2, self.sigma, self.thresholds, True, False, self.medianThresholds,
                                     self.medianRadius, True,
                                     in_channel_min_distance=self.in_channel_min_distance,
                                     between_channel_max_distance=self.between_channel_max_distance)

        if self.verbose:
            print(self.__class__.__name__ + ': found {} spot pairs. pixel coordinates:'.format(len(pairsRaw)))
            for pr in pairsRaw:
                print(pr)

        # plot
        if self.plotDetections:
            self.doPlot([list((np.array(p[0]) + np.array(p[1]))/2) for p in pairsRaw], stack1, stack2)

        ignore_dim = np.array([d for d in stack1.shape][-1::-1]) == 1

        corrected_1 = self.correctForOffset([p[0] for p in pairsRaw], setts, ignore_dim)
        corrected_2 = self.correctForOffset([p[1] for p in pairsRaw], setts, ignore_dim)

        corrected = list(zip(corrected_1, corrected_2))

        if self.verbose:
            print(self.__class__.__name__ + ': found {} spot pairs. offsets:'.format(len(pairsRaw)))
            for pc in corrected:
                print(pc)

        return corrected

class ZDCSpotPairFinder(LegacySpotPairFinder):


    def correctForOffset(self, pairsPixel, setts, ignore_dim):
        offsOld = np.array([get_path_from_dict(
            setts, 'ExpControl/scan/range/{}/off'.format(c), False) for c in ['x', 'y', 'z']], dtype=float)

        # we use the coarse offset here
        offsOld[2] = get_path_from_dict(setts, 'ExpControl/scan/range/coarse_z/g_off', False)

        print(offsOld)
        lensOld = np.array([get_path_from_dict(
            setts, 'ExpControl/scan/range/{}/len'.format(c), False) for c in ['x', 'y', 'z']], dtype=float)

        print(lensOld)
        pszOld = np.array([get_path_from_dict(
            setts, 'ExpControl/scan/range/{}/psz'.format(c), False) for c in ['x', 'y', 'z']], dtype=float)

        print(pszOld)
        
        res = []
        for pair in pairsPixel:
            pairT = np.array(pair, dtype=float)
            #res.append(list(offsOld - (lensOld / 2) + pairT * pszOld))
            res.append(pixel_to_physical_coordinates(pairT, offsOld, lensOld, pszOld, ignore_dim))
        return res


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

    img = np.zeros((1, 1, 201, 201), dtype=float)
    img[0, 0, 100, 100] = 1
    img[0, 0, 20, 50] = 1

    off = [0, 0, 0]
    pixel_size = [0.1, 0.1, 0.1]
    settings_call = ValuesToSettingsDictCallback(lambda: ((off, pixel_size),),(OFFSET_SCAN_PARAMETERS, PIXEL_SIZE_PARAMETERS))
    measurement_settings, hardware_settings = settings_call()[0][0]


    data = MeasurementData()
    data.append(hardware_settings, measurement_settings, [img])
    data_call = lambda: data

    detector = SimpleSingleChannelSpotDetector(data_call, 1, 0.1, verbose=True, plot_detections=True, return_parameter_dict=False)
    detector.invert_dimensions = (False, False, True)
    # res = detector()
    res = ParameterValuesRepeater(SimpleManualOffset(detector, [13,13,13]), 2, nested=True)()
    pprint(res)
