import numpy as np
from scipy import ndimage as ndi
from matplotlib import pyplot as plt
from skimage.color import label2rgb
from skimage.filters import sobel
from skimage.exposure import rescale_intensity, adjust_gamma
from skimage.measure import regionprops
from skimage.morphology import erosion, disk, dilation, square
from skimage.segmentation import clear_border
from skimage.transform import rescale, resize
from sklearn.cluster import KMeans

try:
    # try to import Cellpose, but do not make hard dependency
    from cellpose.models import CellposeModel
except:
    pass

try:
    # try to import StarDist, but do not make hard dependency
    from stardist.models import StarDist2D
    from csbdeep.utils import normalize
except:
    pass

from calmutils.localization import refine_point
from calmutils.misc import filter_rprops

from pipeline2.utils.dict_utils import get_path_from_dict
from pipeline2.detection.spot_detection import pixel_to_physical_coordinates
from pipeline2.utils.parameter_constants import (OFFSET_SCAN_PARAMETERS, OFFSET_STAGE_GLOBAL_PARAMETERS,
                                                 PIXEL_SIZE_PARAMETERS, FOV_LENGTH_PARAMETERS)


def nnet_seg_outer(img, seg_fun, scale_factor=0.5, axis=0, regionprops_filters=None, do_plot=False, ignore_border=True, bg_val=None):

    # default: no regionprops filter
    if regionprops_filters is None:
        regionprops_filters = {}

    # do MIP and segment using StarDist
    mip = img.max(axis=axis)

    # predict on rescaled mip
    original_shape = mip.shape
    mip_sc = rescale(mip, scale_factor, clip=False, preserve_range=True)
    seg = seg_fun(mip_sc)
    seg = resize(seg, original_shape, anti_aliasing=False, order=0, preserve_range=True)

    # ignore objects touching the border
    if ignore_border:
        seg = clear_border(seg)

    # remove objects touching the bg_val area
    if bg_val is not None:
        # we dilate the bg_val area by one pixel and the get all unique labels in that area
        bg_dil_mask = dilation(mip==bg_val, square(3))
        labels_touching_bg = set(list(np.unique(seg[bg_dil_mask])))

        # new mask, setting only objects with labels not in set of background-touching labels
        for rprop in regionprops(seg):
            if rprop.label in labels_touching_bg:
                (min_row, min_col, max_row, max_col) = rprop.bbox
                seg[min_row:max_row, min_col:max_col][rprop.image]=0

    labels = np.zeros_like(seg, dtype=np.int64)
    for idx, rprop in enumerate(regionprops(seg)):
        if filter_rprops(rprop, regionprops_filters):
            (min_row, min_col, max_row, max_col) = rprop.bbox
            labels[min_row:max_row, min_col:max_col][rprop.filled_image] = idx + 1

    if do_plot:
        plt.figure()
        plt.imshow(label2rgb(labels, rescale_intensity(mip)))
        plt.show()

    # cut out objects along z, find brightest plane in (smoothed) z-profile
    res = []
    img_reordered = np.transpose(img, [axis] + [ax for ax in range(len(img.shape)) if ax != axis])
    for rprop in regionprops(labels):
        (min_row, min_col, max_row, max_col) = rprop.bbox
        cut = img_reordered[:, min_row:max_row, min_col:max_col][:, rprop.filled_image]
        sums = np.apply_along_axis(np.sum, 1, cut)
        max_z = np.argmax(ndi.gaussian_filter(sums, 2, mode='constant'))
        max_z_refined = refine_point(ndi.gaussian_filter(sums, 2, mode='constant'), [max_z])

        res.append((min_row, min_col, max_row, max_col, max_z_refined[0]))

    return res


def stardist_midplane_detection(img, model, scale_factor=0.5, prob_tresh=0.8, axis=0, flt=None, do_plot=False, ignore_border=True, bg_val=None):

    def stardist_seg_fun(img):
        seg, _ = model.predict_instances(normalize(img), prob_tresh=prob_tresh)
        return seg

    return nnet_seg_outer(img, stardist_seg_fun, scale_factor, axis, flt, do_plot, ignore_border, bg_val)


def cellpose_midplane_detection(img, model, scale_factor=0.5, flow_tresh=0.2, diameter=50, axis=0, flt=None, do_plot=False, ignore_border=True, bg_val=None):

    def cellpose_seg_fun(img):
        segs, *_ = model.eval([img], flow_threshold=flow_tresh, diameter=diameter, channels=[0,0])
        return segs[0]

    return nnet_seg_outer(img, cellpose_seg_fun, scale_factor, axis, flt, do_plot, ignore_border, bg_val)


def nucleus_midplane_detection(img, axis=0, flt=None, do_plot=False, ignore_border=True, bg_val=None, n_classes=2):
    """
    Detect midplanes of nuclei in image stack

    Parameters
    ----------
    img: 3d-array
        the image
    axis: int, optional
        index of the z-axis
    flt: dict, optional
        dict of filters of the form 'rprop_name' : (min, max)
        only objects whose regionprops fall within the range for all filters will be considered
    do_plot: boolean
        whether to plot the segmentation or not
    ignore_border: boolean
        whether to ignore object on the image border or not
    bg_val: numeric
        value of empty background (e.g. in stitched images), will be ignored in segmentation

    Returns
    -------
    midplanes: list of tuples
        midplanes of the detected nuclei as tuples:
        (min_row, min_col, max_row, max_col, max_z_refined)
    """

    # TODO? make sigmas, gamma, disk radius user-settable?

    # default: no filter
    if flt is None:
        flt = {}

    # max project along z
    # use gamma corrected mip, blur and edge images as features
    mip = np.max(img, axis=axis)
    if bg_val is not None:
        mip[mip==bg_val] = 0
    mip = rescale_intensity(mip)
    mip = adjust_gamma(mip, 0.5)
    blur = ndi.gaussian_filter(mip, 3)
    edge = sobel(mip)

    # 2-means clustering of features for segmentation
    if bg_val is None:
        feat = np.dstack([mip, blur, edge])
        feat = feat.reshape((np.prod(feat.shape[:-1]), feat.shape[-1]))
        km = KMeans(n_classes, n_init='auto')
        seg = km.fit_predict(feat).reshape(mip.shape)
    else:
        mip2 = np.apply_along_axis(np.max, axis, img)
        test = mip[mip2 != bg_val]
        print(test.shape)
        feat = np.dstack([mip[mip2 != bg_val], blur[mip2 != bg_val], edge[mip2 != bg_val]])
        print(feat.shape)
        feat = feat.reshape((-1,3))
        km = KMeans(n_classes, n_init='auto')
        seg_tmp = km.fit_predict(feat)
        seg = np.zeros_like(mip)
        seg[mip2 != bg_val] = seg_tmp

    # k-Means might arbitrarily number classes, pick the brightest as foreground
    max_class = np.argmax([np.mean(mip[seg==i]) for i in range(n_classes)])
    seg = (seg == max_class ) * 1

    # ignore objects touching the border
    if ignore_border:
        seg = clear_border(seg)
        
    # remove objects touching the bg_val area
    if bg_val is not None:
        mip2 = np.max(img, axis=axis)
        # we dilate the bg_val area by one pixel and the get all unique labels in that area
        bg_dil_mask = dilation(mip2==bg_val, square(3))
        labels, n_objs = ndi.label(seg)
        labels_touching_bg = set(list(np.unique(labels[bg_dil_mask])))

        # new mask, setting only objects with labels not in set of background-touching labels
        seg2 = np.zeros_like(seg)
        for rprop in regionprops(labels):
            if not rprop.label in labels_touching_bg:
                (min_row, min_col, max_row, max_col) = rprop.bbox
                seg2[min_row:max_row, min_col:max_col][rprop.image]=1
        seg = seg2

    # cleanup labels (erosion, size filtering)
    seg = erosion(seg, disk(3))
    labels, n_objs = ndi.label(seg)
    labels2 = np.zeros_like(labels, dtype=np.int64)
    for idx, rprop in enumerate(regionprops(labels)):
        if filter_rprops(rprop, flt):
            (min_row, min_col, max_row, max_col) = rprop.bbox
            labels2[min_row:max_row, min_col:max_col][rprop.filled_image] = idx + 1

    # do not relabel for now, we do not need it and
    # a UserWarning is raised here, caused by skimage
    # labels, _, _ = relabel_sequential(labels2)
    labels = labels2

    if do_plot:
        plt.figure()
        plt.imshow(label2rgb(labels, rescale_intensity(mip)))
        plt.show()

    # cut out objects along z, find brightest plane in (smoothed) z-profile
    res = []
    img_reordered = np.transpose(img, [axis] + [ax for ax in range(len(img.shape)) if ax != axis])
    for rprop in regionprops(labels):
        (min_row, min_col, max_row, max_col) = rprop.bbox
        cut = img_reordered[:, min_row:max_row, min_col:max_col][:, rprop.filled_image]
        sums = np.apply_along_axis(np.sum, 1, cut)
        max_z = np.argmax(ndi.gaussian_filter(sums, 2, mode='constant'))
        max_z_refined = refine_point(ndi.gaussian_filter(sums, 2, mode='constant'), [max_z])

        res.append((min_row, min_col, max_row, max_col, max_z_refined[0]))

    return res


class SimpleNucleusMidplaneDetector:

    def __init__(self, data_source_callback, configuration=0, channel=0, n_classes=2, manual_offset=0, use_stage=False,
                 region_filters={}, fov_expansion_factor=1.2, plot_detections=False, verbose=False):
        self.data_source_callback = data_source_callback
        self.configuration = configuration
        self.channel = channel
        self.verbose = verbose
        self.do_plot = plot_detections
        self.region_filters = region_filters
        self.expand = fov_expansion_factor
        self.n_classes = n_classes
        self.manual_offset = manual_offset
        self.use_stage = use_stage

    def midplane_detection_fun(self, img):
        return nucleus_midplane_detection(img, 0, self.region_filters, self.do_plot, True, -1, self.n_classes)

    def __call__(self):

        data = self.data_source_callback()

        # no data yet -> empty update
        if data is None:
            if self.verbose:
                print(self.__class__.__name__ + ': ERROR: no image for nucleus detection')
            return []

        if (data.num_configurations <= self.configuration) or (data.num_channels(self.configuration) <= self.channel):
            raise ValueError('no images present. TODO: fail gracefully/skip here')

        img = data.data[self.configuration][self.channel][0, :, :, :]
        img = np.array(img, float)
        
        # check which dimensions are singleton
        ignore_dimensions = np.array([d for d in img.shape]) == 1

        # get position / FOV parameters for measurement we detect in
        settings = data.measurement_settings[self.configuration]
        if self.use_stage:
            offsets_old = np.array([get_path_from_dict(settings, path, False) for path in OFFSET_STAGE_GLOBAL_PARAMETERS], dtype=float)
        else:
            offsets_old = np.array([get_path_from_dict(settings, path, False) for path in OFFSET_SCAN_PARAMETERS], dtype=float)
        fov_lengths_old = np.array([get_path_from_dict(settings, path, False) for path in FOV_LENGTH_PARAMETERS], dtype=float)
        pixel_sizes_old = np.array([get_path_from_dict(settings, path, False) for path in PIXEL_SIZE_PARAMETERS], dtype=float)

        if self.verbose:
            print('Nucleus Detection:')
            print('old offset: {}'.format(offsets_old))
            print('old len: {}'.format(fov_lengths_old))
            print('old psz: {}'.format(pixel_sizes_old))

        midplanes = self.midplane_detection_fun(img)

        res = []
        for (ymin, xmin, ymax, xmax, zmid) in midplanes:

            if self.verbose:
                print('pixel result: {}'.format((ymin, xmin, ymax, xmax, zmid)))
            
            # get offset and fov in world units
            off = np.array([zmid, (ymax+ymin)/2, (xmax+xmin)/2], dtype=float)
            off = pixel_to_physical_coordinates(off, offsets_old, fov_lengths_old, pixel_sizes_old, ignore_dimensions)
            fov = [None, (ymax - ymin)*pixel_sizes_old[1]*self.expand, (xmax - xmin)*pixel_sizes_old[2]*self.expand]
            
            off[0] += self.manual_offset

            if self.verbose:
                print(self.__class__.__name__ + ': Found Nucleus at {}, FOV: {}'.format(off, fov))

            res.append((list(off), fov))

        return res


class StarDistNucleusMidplaneDetector(SimpleNucleusMidplaneDetector):
    def __init__(self, data_source_callback, scale_factor=0.5, prob_thresh=0.8, **kwargs):

        super().__init__(data_source_callback, **kwargs)
        self.n_classes = 0

        # TODO: make settable?
        self.pretrained_model_id = '2D_versatile_fluo'

        self.scale_factor = scale_factor
        self.prob_tresh = prob_thresh
        self.model = StarDist2D(self.pretrained_model_id)

    def midplane_detection_fun(self, img):
        return stardist_midplane_detection(img, self.model, self.scale_factor, self.prob_tresh, 0, self.region_filters, self.do_plot, True, -1)


class CellposeNucleusMidplaneDetector(SimpleNucleusMidplaneDetector):
    def __init__(self, data_source_callback, scale_factor=0.5, flow_thresh=0.2, diameter=50, **kwargs):

        super().__init__(data_source_callback, **kwargs)
        self.n_classes = 0

        self.scale_factor = scale_factor
        self.flow_tresh = flow_thresh
        self.diameter = diameter

        self.model = CellposeModel(gpu=True, model_type='nuclei')

    def midplane_detection_fun(self, img):
        return cellpose_midplane_detection(img, self.model, self.scale_factor, self.flow_tresh, self.diameter, 0, self.region_filters, self.do_plot, True, -1)

if __name__ == '__main__':

    from pipeline2.data import MeasurementData
    from pprint import pprint
    import logging
    from pipeline2.taskgeneration.coordinate_building_blocks import ValuesToSettingsDictCallback
    from skimage.io import imread

    img = imread('/Users/david/Downloads/dapi_nuclei.tif')
    img = img.reshape((1,1,img.shape[0],img.shape[1]))

    logging.basicConfig(level=logging.INFO)

    off = [0, 0, 0]
    pixel_size = [0.1, 0.1, 0.1]
    fov = np.array([0.1, 0.1, 0.1]) * np.array(img.shape[1:])
    settings_call = ValuesToSettingsDictCallback(lambda: ((off, pixel_size, fov),),
                                                 (OFFSET_SCAN_PARAMETERS, PIXEL_SIZE_PARAMETERS, FOV_LENGTH_PARAMETERS))

    measurement_settings, hardware_settings = settings_call()[0][0]

    data = MeasurementData()
    data.append(hardware_settings, measurement_settings, [img])
    data_call = lambda: data

    detector = SimpleNucleusMidplaneDetector(data_call, plot_detections=True)
    detector = CellposeNucleusMidplaneDetector(data_call, diameter=20, plot_detections=True, manual_offset=1)
    # res = detector()

    from pipeline2.taskgeneration.coordinate_building_blocks import ScanFieldSettingsGenerator
    res = ScanFieldSettingsGenerator(detector, True)()
    pprint(res)
