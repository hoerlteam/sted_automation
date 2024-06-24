from operator import add
from functools import reduce

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import to_rgb
from matplotlib.colors import get_named_colors_mapping


# use the same colors as ImageJ Merge Channels by default
DEFAULT_COLOR_NAMES = ('red', 'green', 'blue', 'gray', 'cyan', 'magenta', 'yellow')


def grayscale_to_named_color(img_grayscale, color_name):

    # get normalized RGB values of named color
    color = np.array(to_rgb(get_named_colors_mapping()[color_name]))
    color /= np.max(color)

    # normalize image
    img_grayscale = img_grayscale / np.max(img_grayscale)

    # to (... original shape ..., 3)-shape RGB via outer product
    img_color = np.outer(img_grayscale, color).reshape(img_grayscale.shape + (-1,))
    return img_color


def gray_images_to_composite(images, color_names=DEFAULT_COLOR_NAMES):

    # check that images have the same shape
    shape = None
    for image in images:
        if shape is not None and image.shape != shape:
            raise ValueError('Images must have the same shape')
        shape = image.shape

    # make channels, compose via add, clip overflowing values
    composite_channels = [grayscale_to_named_color(image, color_name) for image, color_name in zip(images, color_names)]
    composite = np.clip(reduce(add, composite_channels), 0, 1)

    return composite


def make_projection(img, axis=0, fun=np.max):
    return fun(img, axis=axis)


def normalize(arr, intensity_range=None):
    if intensity_range is not None:
        intensity_range_min = intensity_range[0]
        intensity_range_max = intensity_range[1]
        arr1 = (arr - intensity_range_min) / (intensity_range_max - intensity_range_min)
        return np.clip(arr1, 0.0, 1.0)
    else:
        return (arr - np.min(arr)) / (np.max(arr) - np.min(arr))


def make_rgb_max_projection(images, normalization_range=None, axis=None, percentile_range=False, color_names=DEFAULT_COLOR_NAMES):
    """
    Parameters:
    ===========
    percentile_range: boolean
        Whether to interpret the display range ran as percentiles or raw min & max intensity to display (default)
    """

    # max project all but the last two dimensions by default
    if axis is None:
        axis = tuple(np.arange(images[0].ndim - 2, dtype=int))

    projections = [make_projection(img, axis) for img in images]

    # get percentiles for normalization
    if percentile_range and normalization_range is not None:
        ranges = [np.percentile(img, normalization_range) for img in projections]
        print(ranges)
    else:
        ranges = [normalization_range] * len(projections)
    projections = [normalize(img, range_i) for img, range_i in zip(projections, ranges)]

    return gray_images_to_composite(projections, color_names=color_names)


def draw_detections_multicolor(images, coordinates, normalization_range=None, axis=None, size=3, percentile_range=False,
                               color_names=DEFAULT_COLOR_NAMES, marker_color='white'):
    fig, ax = plt.subplots()
    rgb_image = make_rgb_max_projection(images, normalization_range, axis, percentile_range, color_names)
    ax.imshow(rgb_image)

    if len(coordinates) == 0:
        fig.show()
        return

    coordinates = np.array(coordinates)

    if axis is None:
        axis = tuple(np.arange(coordinates.shape[1] - 2, dtype=int))

    coordinates = coordinates.T[~np.isin(np.arange(coordinates.shape[1]), axis)].T

    for d in coordinates:
        c = plt.Circle((d[1], d[0]), size, color=marker_color, linewidth=1.5, fill=False)
        ax.add_patch(c)

    fig.show()


def draw_detections_1c(im, detections, ran=None, projection_axis=0, siz=3, percentile_range=False):
    draw_detections_multicolor([im], detections, ran, projection_axis, siz, percentile_range,
                               color_names=['gray'], marker_color='red')
