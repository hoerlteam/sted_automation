from skimage.feature import blob_dog, blob_log, blob_doh
from scipy import ndimage, spatial, stats
import numpy as np
import skimage
from matplotlib import pyplot as plt
import re
import os
from collections import defaultdict
from csv import DictReader


def make_proj(img, axis=0, fun=np.max):
    return np.apply_along_axis(fun, axis, img)


def normalize(arr, ran=None):
    if ran is not None:
        arr1 = (arr - ran[0]) / (ran[1] - ran[0])
        arr1[arr1 < 0] = 0.0
        arr1[arr1 > 1] = 1.0
        return arr1
    else:
        return (arr - np.min(arr)) / (np.max(arr) - np.min(arr))


def make_rgb_maxproj(im1, im2, ran=None, axis=None, percentile_range=False):
    '''
    TODO: documentation
    Parameters:
    ===========
    percentile_range: boolean
        Whether to interpret the display range ran as percentiles or raw min & max intensity to display (default)
    '''
    if axis != None:
        p_im1 = make_proj(im1, axis)
        p_im2 = make_proj(im2, axis)
    else:
        p_im1 = make_proj(im1, len(im1.shape) - 1)
        p_im2 = make_proj(im2, len(im2.shape) - 1)

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


def draw_detections_1c(im, dets, ran=None, axis=None, siz=3, percentile_range=False):
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    im1 = make_proj(im, axis if axis is not None else len(im.shape) - 1)

    if percentile_range and ran is not None:
        ran = np.percentile(im1, ran)

    im1 = normalize(im1, ran)
    plt.imshow(im1, cmap='gray')
    if axis == None:
        axis = len(im.shape) - 1
    for d in dets:
        d1 = np.array(d)[np.arange(3) != axis]
        c = plt.Circle((d1[1], d1[0]), siz, color='red', linewidth=1.5, fill=False)
        ax.add_patch(c)
    plt.draw()


def read_analysis_results(path):
    res = defaultdict(list)
    with open(path, 'r') as fd:
        dr = DictReader(fd)

        for line in dr:
            res[line['file'].split(os.path.sep)[-1]].append(
                (float(line['d11']), float(line['d01']), float(line['d21'])))
            res[line['file'].split(os.path.sep)[-1]].append(
                (float(line['d12']), float(line['d02']), float(line['d22'])))

    #print(res)
    return res


def plot_analysis_results(path, csvpath, ran, pix_siz=0.02):
    p = '(.*?)(ch[0-9]+?\.tif)'

    acquisitions = defaultdict(list)
    res = read_analysis_results(csvpath)

    for f in next(os.walk(path))[2]:
        m = re.match(p, f)
        if m:
            acquisitions[m.groups()[0]].append(m.groups()[1])
            acquisitions[m.groups()[0]].sort()

    for k, v in acquisitions.items():
        print('--- ' + k + ' ---')
        im0 = np.array(read_image_stack(os.path.join(path, k) + v[0], True), np.float)
        im1 = np.array(read_image_stack(os.path.join(path, k) + v[1], True), np.float)

        dets = res[k]
        dets2 = list()
        for d in dets:
            dets2.append(np.array(d) / pix_siz)

        print('--- found ' + str(len(dets)) + ' pairs')

        draw_detections_2c(im0, im1, dets2, ran)
        plt.show()


def plot_files(path, ran=None, thresh=0.1, sigma=3):
    p = '(.*?)(ch[0-9]+?\.tif)'

    acquisitions = defaultdict(list)

    for f in next(os.walk(path))[2]:
        m = re.match(p, f)
        if m:
            acquisitions[os.path.join(path, m.groups()[0])].append(m.groups()[1])
            acquisitions[os.path.join(path, m.groups()[0])].sort()

    for k, v in acquisitions.items():
        print('--- ' + k + ' ---')
        im0 = np.array(read_image_stack(k + v[0], True), np.float)
        im1 = np.array(read_image_stack(k + v[1], True), np.float)

        dets = pair_finder_inner(im0, im1, sigma, thresh, False, False)

        print('--- found ' + str(len(dets)) + ' pairs')

        draw_detections_2c(im0, im1, dets, ran)
        plt.show()


def main():
    # plt.imshow(make_proj(read_image_stack( '/Users/david/Desktop/ov_ch1.tif'  ), 2))
    # plt.show()

    im0 = read_image_stack('/Users/david/Desktop/ov_ch0.tif', True)
    im1 = read_image_stack('/Users/david/Desktop/ov_ch1.tif', True)

    print(im0)

    draw_detections_1c(im0, [[20, 12, 12], [40, 40, 12]])
    draw_detections_2c(im0, im1, [[20, 12, 12], [40, 40, 12]], [0.5, 99.99], percentile_range=True)

    plt.show()


if __name__ == '__main__':
    # print(read_analysis_results('/Users/david/Desktop/AutomatedAcquisitions/out.csv'))
    main()
