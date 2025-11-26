from operator import attrgetter

import numpy as np
from scipy.ndimage import gaussian_filter
from skimage.morphology import label, convex_hull_image, binary_dilation, binary_erosion
from skimage.segmentation import morphological_geodesic_active_contour
from skimage.filters import threshold_otsu
from skimage.measure import regionprops

from calmutils.morphology.structuring_elements import ellipsoid_selem



def keep_only_largest_component(mask, in_place=True):

    """
    Set all connected components in binary mask (or label image) but the largest to 0.
    """

    # make copy or modify in place
    res = mask if in_place else mask.copy()

    # label, get regionprops and sort by area/volume (descending)
    labels = label(mask)
    rprops = regionprops(labels)
    rprops.sort(key=attrgetter('area'), reverse=True)

    # set all but the first (largest) to 0
    for rp in rprops[1:]:
        res[rp.slice][rp.image] = 0

    return res


def get_outline_mask(mask, expand_innner=1, expand_outer=0):

    """
    Get outlines of a binary mask via xor of eroded and dilated version.
    Expansion radii can be given as scalar or for each dimension individually.
    """

    # scalar expansion: same along all dimensions
    if np.isscalar(expand_innner):
        expand_innner = [expand_innner] * mask.ndim
    if np.isscalar(expand_outer):
        expand_outer = [expand_outer] * mask.ndim

    # ensure array
    expand_innner, expand_outer = np.array(expand_innner), np.array(expand_outer)

    # if both inner and outer expansion are 0, we can't get a meaningful result
    if np.any(expand_innner == 0) and np.any(expand_outer == 0):
        raise ValueError('either inner or outer expansion must be > 0.')

    # ensure mask is binary
    mask = mask > 0

    # erode mask (or keep for no inner expansion)
    if np.all(expand_innner > 0):
        mask_eroded = binary_erosion(mask, ellipsoid_selem(expand_innner))
    else:
        mask_eroded = mask

    # dilate mask (or keep for no outer expansion)
    if np.all(expand_outer > 0):
        mask_dilated = binary_dilation(mask, ellipsoid_selem(expand_outer))
    else:
        mask_dilated = mask

    # outline is xor of eroded and dialted
    outlines = mask_eroded ^ mask_dilated
    return outlines


def periphery_segmentation_active_contours(img, blur_sigma=2, active_contour_iterations=40, return_outlines=True, outline_radius=2):

    """
    Simple segmentation of image with object border stained (e.g. nucleoporins, lamin for nucleus).
    Will first perform Otsu thresholding and then snap the mask to intensity image via morphological active countours.
    NOTE: This assumes a single object in the input.
    """

    # blur and threshold
    g = gaussian_filter(img.astype(float), blur_sigma)
    mask = g > threshold_otsu(g)

    # discard all but largest object and get convex hull
    mask = keep_only_largest_component(mask)
    mask = convex_hull_image(mask)

    # refine mask (snap to edge) via active contour
    # AC snaps to low values, so we pass the inverted gaussian filtered image
    mask = morphological_geodesic_active_contour(-g, active_contour_iterations, init_level_set=mask)

    # return mask
    if not return_outlines:
        return mask
    # or: calculate outlines and return
    else:
        outlines = get_outline_mask(mask, expand_innner=outline_radius, expand_outer=outline_radius)
        return outlines