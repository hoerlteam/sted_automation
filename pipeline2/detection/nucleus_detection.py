import numpy as np
from scipy import ndimage as ndi
from matplotlib import pyplot as plt
from skimage.color import label2rgb
from skimage.filters import sobel
from skimage.exposure import rescale_intensity, adjust_gamma
from skimage.measure import regionprops
from skimage.morphology import erosion, disk
from skimage.segmentation import relabel_sequential, clear_border
from sklearn.cluster import KMeans

from calmutils.localization import refine_point
from calmutils.misc import filter_rprops

from ..util import filter_dict
from .detection import _correct_offset


def nucleus_midplane_detection(img, axis=0, flt=None, do_plot=False, ignore_border=True):
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
    mip = np.apply_along_axis(np.max, axis, img)
    mip = rescale_intensity(mip)
    mip = adjust_gamma(mip, 0.5)
    blur = ndi.gaussian_filter(mip, 3)
    edge = sobel(mip)

    # 2-means clustering of features for segmentation
    feat = np.dstack([mip, blur, edge])
    feat = feat.reshape((np.prod(feat.shape[:-1]), feat.shape[-1]))
    km = KMeans(2)
    seg = km.fit_predict(feat).reshape(mip.shape)

    # k-Means might call backround 0 and foreground 1
    if (np.mean(mip[seg == 0]) > np.mean(mip[seg == 1])):
        seg = (seg - 1) * -1

    # ignore objects touching the border
    if ignore_border:
        seg = clear_border(seg)

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


class SimpleNucleusMidplaneDetector():

    def __init__(self, dataSource, configuration=0, channel=0):
        self.dataSource = dataSource
        self.configuration = configuration
        self.channel = channel
        self.verbose = False
        self.do_plot = False
        self.filt = {}
        self.expand = 1.2

    def withVerbose(self, verbose=True):
        self.verbose = verbose
        return self

    def withPlot(self, plot=True):
        self.do_plot = plot
        return self

    def withFilter(self, filt=True):
        self.filt = filt
        return self
    
    def withFOVExpansion(self, expand):
        self.expand = expand
        return self

    def get_fields(self):

        data = self.dataSource.get_data()

        # no data yet -> empty update
        if data is None:
            if self.verbose:
                print(self.__class__.__name__ + ': ERROR: no image for nucleus detection')
            return []

        if (data.numConfigurations <= self.configuration) or (data.numImages(self.configuration) <= self.channel):
            raise ValueError('no images present. TODO: fail gracefully/skip here')

        img = data.data[self.configuration][self.channel][0, :, :, :]
        # make float
        img = np.array(img, np.float)
        
        # check which dimensions are singleton (note: x,y,z here!)
        ignore_dim = np.array([d for d in img.shape][-1::-1]) == 1
        
        setts = data.measurementSettings[self.configuration]

        offsOld = np.array([filter_dict(
            setts, 'ExpControl/scan/range/{}/off'.format(c), False) for c in ['x', 'y', 'z']], dtype=float)

        lensOld = np.array([filter_dict(
            setts, 'ExpControl/scan/range/{}/len'.format(c), False) for c in ['x', 'y', 'z']], dtype=float)

        pszOld = np.array([filter_dict(
            setts, 'ExpControl/scan/range/{}/psz'.format(c), False) for c in ['x', 'y', 'z']], dtype=float)


        if self.verbose:
            print('Nucleus Detection:')
            print('old offset: {}'.format(offsOld))
            print('old len: {}'.format(lensOld))
            print('old psz: {}'.format(pszOld))
            
        midplanes = nucleus_midplane_detection(img, 0, self.filt, self.do_plot, True)

        res = []

        for (ymin, xmin, ymax, xmax, zmid) in midplanes:

            if self.verbose:
                print('pixel result: {}'.format((ymin, xmin, ymax, xmax, zmid)))
            
            # get offset and fov in world units
            off = np.array([(xmax+xmin)/2, (ymax+ymin)/2, zmid], dtype=np.float)
            
            if self.verbose:
                print('pixel off: {}'.format(off))
                
            off = _correct_offset(off, offsOld, lensOld, pszOld, ignore_dim)
            fov = [(xmax - xmin)*pszOld[0]*self.expand, (ymax - ymin)*pszOld[1]*self.expand, None]

            if self.verbose:
                print(self.__class__.__name__ + ': Found Nucleus at {}, FOV: {}'.format(off, fov))

            res.append((list(off), fov))

        return res
