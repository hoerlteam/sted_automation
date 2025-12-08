import numpy as np
from scipy.ndimage import gaussian_filter
from skimage.morphology import convex_hull_image
from skimage.segmentation import morphological_geodesic_active_contour
from skimage.filters import threshold_otsu

from calmutils.morphology.structuring_elements import ellipsoid_selem
from calmutils.morphology.mask_postprocessing import keep_only_largest_component, get_mask_outlines


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
        outlines = get_mask_outlines(mask, expand_innner=outline_radius, expand_outer=outline_radius)
        return outlines


def get_axis_aligned_poles(mask):

    """
    Get "poles" of a mask image along all axes
    i.e. coordinates of the first and last nonzero pixel along each axis
    """

    nonzero_coords = np.stack(np.nonzero(mask)).T

    # add min and max coord index for each dimension
    pole_idxs = []
    for d in range(mask.ndim):
        pole_idxs.append(np.argmin(nonzero_coords.T[d]))
        pole_idxs.append(np.argmax(nonzero_coords.T[d]))

    return nonzero_coords[pole_idxs]
