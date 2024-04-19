import numpy as np
import re
import os
from collections import defaultdict
from csv import DictReader

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