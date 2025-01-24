import logging

from scipy import ndimage, spatial
import numpy as np
from skimage.feature import peak_local_max

logger = logging.getLogger(__name__)


def cleanup_kdtree(img, kdt, dets, dist):
    notdone = True
    while notdone:
        notdone = False
        for i in range(len(dets)):
            # find nearest neighbor in the same channel, if it is closer than dist, keep the brighter of the two
            nn = kdt.query(dets[i], k=2, distance_upper_bound=dist)
            if (not np.isinf(nn[0][1])) and nn[0][1] > 0:
                ineighbor = nn[1][1]
                if img[tuple(dets[i])] < img[tuple(dets[ineighbor])]:
                    dets.pop(i)
                else:
                    dets.pop(ineighbor)
                notdone = True

                # rebuild kdtree
                kdt = spatial.KDTree(dets)
                break


def find_pairs(
    kdt2, dets1, dets2, dist, invert_axes=True, center=True, return_pair=False
):
    res = []
    for d in dets1:
        # find nearest neighbor in channel 2, if it is closer than dist
        nn = kdt2.query(d, distance_upper_bound=dist)
        if not (np.isinf(nn[0]) or nn[1] >= len(dets2)):
            # dets were in zyx -> turn to xyz
            if invert_axes:
                if return_pair:
                    res.append((d[-1::-1], dets2[nn[1]][-1::-1]))
                else:
                    res.append(
                        (np.array(d[-1::-1]) + np.array(dets2[nn[1]][-1::-1])) / 2
                        if center
                        else d[-1::-1]
                    )
            else:
                if return_pair:
                    res.append((d, dets2[nn[1]]))
                else:
                    res.append(
                        (np.array(d) + np.array(dets2[nn[1]])) / 2 if center else d
                    )
    return res


def detect_blobs_find_pairs(
    stack1,
    stack2,
    pix_sig,
    threshold,
    invert_axes,
    normalize,
    median_thresholds,
    median_radius,
    return_pair=False,
    in_channel_min_distance=3,
    between_channel_max_distance=5,
):

    # detect blobs via Laplacian-of-Gaussian (only blobs brighter than threshold)
    sig = pix_sig / np.sqrt(2)
    dets1 = detect_blobs(
        stack1,
        [sig, sig, sig],
        threshold[0],
        normalize,
        median_thresholds[0],
        median_radius,
    )
    dets1 = list(dets1)
    dets2 = detect_blobs(
        stack2,
        [sig, sig, sig],
        threshold[1],
        normalize,
        median_thresholds[1],
        median_radius,
    )
    dets2 = list(dets2)

    logger.debug("no of candidates stack1: {}".format(len(dets1)))
    logger.debug("no of candidates stack2: {}".format(len(dets2)))

    # did not find any spots in one of the channels -> return empty results
    if len(dets1) == 0 or len(dets2) == 0:
        return []

    # put spots in kd-tree for fast nearesr neighbor calculations
    kd1 = spatial.KDTree(dets1)
    kd2 = spatial.KDTree(dets2)
    # if spots (in one channel) are closer than in_channel_min_distance pixels, pick the brighter of the two
    cleanup_kdtree(stack1, kd1, dets1, in_channel_min_distance)
    cleanup_kdtree(stack2, kd2, dets2, in_channel_min_distance)

    logger.debug("remaining after cleanup stack1: {}".format(len(dets1)))
    logger.debug("remaining after cleanup stack2: {}".format(len(dets2)))

    # for every remaining spot in image1, return a candidate pair if there is a spot in channel 2
    # that is closer than between_channel_max_distance pixels to it
    res = []
    for p in find_pairs(
        kd2,
        dets1,
        dets2,
        between_channel_max_distance,
        invert_axes,
        return_pair=return_pair,
    ):
        res.append(list(p) if not return_pair else p)
    return res


def detect_blobs(
    img,
    sigmas,
    threshold,
    normalize=False,
    threshold_rel_median=3,
    med_radius=5,
    refine=False,
):

    # make float, but do not normalize / scale
    img = img.astype(float)

    # scale image to 0=min 1=max
    if normalize:
        img = (img - np.min(img)) / (np.max(img) - np.min(img))

    # do LoG filtering and local maxima detection
    log_img = -ndimage.gaussian_laplace(img, sigmas) * (sigmas[0] ** 2)
    peaks = peak_local_max(
        log_img, threshold_abs=threshold, min_distance=1, exclude_border=False
    )

    # if a nonzero threshold_rel_median is set, make sure that
    # blob peak is at least x-fold brighter than median within med_radius
    if threshold_rel_median:
        median_img = ndimage.median_filter(img, med_radius)
        peaks = [
            p
            for p in peaks
            if img[tuple(p)] > median_img[tuple(p)] * threshold_rel_median
        ]

    # refine via quadratic fit
    if refine:
        from calmutils.localization import refine_point

        peaks = [refine_point(img, p) for p in peaks]

    return peaks
