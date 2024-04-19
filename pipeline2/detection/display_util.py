import numpy as np
from matplotlib import pyplot as plt


def make_projection(img, axis=0, fun=np.max):
    return np.apply_along_axis(fun, axis, img)


def normalize(arr, intensity_range=None):
    if intensity_range is not None:
        intensity_range_min = intensity_range[0]
        intensity_range_max = intensity_range[1]
        arr1 = (arr - intensity_range_min) / (intensity_range_max - intensity_range_min)
        return np.clip(arr1, 0.0, 1.0)
    else:
        return (arr - np.min(arr)) / (np.max(arr) - np.min(arr))


def make_rgb_maxproj(im1, im2, ran=None, axis=0, percentile_range=False):
    '''
    TODO: documentation
    Parameters:
    ===========
    percentile_range: boolean
        Whether to interpret the display range ran as percentiles or raw min & max intensity to display (default)
    '''
    p_im1 = make_projection(im1, axis)
    p_im2 = make_projection(im2, axis)

    if percentile_range and ran is not None:
        ran_im1 = np.percentile(p_im1, ran)
        ran_im2 = np.percentile(p_im2, ran)
    else:
        ran_im1, ran_im2 = ran, ran

    return np.dstack((normalize(p_im1, ran_im1), normalize(p_im2, ran_im2), np.zeros(p_im1.shape)))


def draw_detections_2c(im1, im2, dets, ran=None, axis=None, siz=3, percentile_range=False):
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    rgb = make_rgb_maxproj(im1, im2, ran, axis, percentile_range)
    plt.imshow(rgb)
    if axis is None:
        axis = len(im1.shape) - 1
    for d in dets:
        d1 = np.array(d)[np.arange(3) != axis]
        c = plt.Circle((d1[1], d1[0]), siz, color='white', linewidth=1.5, fill=False)
        ax.add_patch(c)
    plt.draw()
    plt.show()


def draw_detections_1c(im, detections, ran=None, projection_axis=0, siz=3, percentile_range=False):
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    im1 = make_projection(im, projection_axis)

    if percentile_range and ran is not None:
        ran = np.percentile(im1, ran)

    im1 = normalize(im1, ran)
    plt.imshow(im1, cmap='gray')

    for d in detections:
        d1 = np.array(d)[np.arange(3) != projection_axis]
        c = plt.Circle((d1[1], d1[0]), siz, color='red', linewidth=1.5, fill=False)
        ax.add_patch(c)
    plt.draw()
