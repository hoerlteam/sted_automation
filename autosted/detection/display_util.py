import numpy as np
from matplotlib import pyplot as plt

from calmutils.color.color import DEFAULT_COLOR_NAMES, gray_images_to_rgb_composite


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

    return gray_images_to_rgb_composite(projections, color_names=color_names)


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

    plt.show()


def draw_detections_1c(im, detections, ran=None, projection_axis=0, siz=3, percentile_range=False):
    draw_detections_multicolor([im], detections, ran, projection_axis, siz, percentile_range,
                               color_names=['gray'], marker_color='red')


def draw_bboxes_multicolor(images, bboxes, normalization_range=None, axis=None, box_width=2, percentile_range=False,
                           color_names=DEFAULT_COLOR_NAMES, box_color='white'):

    # projection axes default: all but last two
    if axis is None:
        axis = tuple(np.arange(images[0].ndim - 2, dtype=int))

    # plot projected image
    fig, ax = plt.subplots()
    rgb_image = make_rgb_max_projection(images, normalization_range, axis, percentile_range, color_names)
    ax.imshow(rgb_image)

    for bbox in bboxes:

        # split min_0, min_1, ..., max_0, max_1, ... bbox into mins and maxs
        mins = np.array(bbox[:len(bbox)//2])
        maxs = np.array(bbox[len(bbox)//2:])
        # drop dimensions that we projected in images
        mins = mins[~np.isin(np.arange(len(mins)), axis)]
        maxs = maxs[~np.isin(np.arange(len(maxs)), axis)]

        # start of bbox in xy <- inversed min coords
        xy_start = mins[::-1]
        # height, width
        h, w = maxs - mins

        # add to plot
        rec = plt.Rectangle(xy_start, w, h, edgecolor=box_color, fill=None, linewidth=box_width)
        ax.add_patch(rec)

    plt.show()


def draw_bboxes_1c(image, bboxes, normalization_range=None, axis=None, box_width=2, percentile_range=False):
    draw_bboxes_multicolor([image], bboxes, normalization_range, axis, box_width, percentile_range,
                           color_names=['gray'], box_color='red')

