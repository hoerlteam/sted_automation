from skimage.feature import blob_dog, blob_log, blob_doh
from scipy import ndimage, spatial, stats
import numpy as np
import skimage


def mean_along_axis(img, axis):
    '''
    calculate mean of every hyper-slice in image along axis
    '''
    axes = tuple([ax for ax in range(len(img.shape)) if ax != axis])
    #print(axes)
    profile = np.mean(img, axes)
    return profile


def focus_in_stack(img, pixsize_z):
    # get mean profile, smooth it via a Gaussian blur
    profile = mean_along_axis(img, 1)
    smoothprofile = ndimage.gaussian_filter1d(profile, 3, mode='constant')
    tmax = np.argmax(smoothprofile)
    # calculate offset of maximum in comparison to middle
    pix_d = tmax - (len(profile) / 2)
    return pix_d * pixsize_z


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
        

def find_pairs(kdt2, dets1, dets2, dist, invertAxes=True, center=True):
    res = []
    for d in dets1:
        # find nearest neighbor in channel 2, if it is closer than dist
        nn = kdt2.query(d, distance_upper_bound=dist)
        if not (np.isinf(nn[0]) or nn[1] >= len(dets2)):
            # dets were in zyx -> turn to xyz
            if invertAxes:
                res.append((np.array(d[-1::-1]) + np.array(dets2[nn[1]][-1::-1])) / 2 if center else d[-1::-1])
            else:
                res.append((np.array(d) + np.array(dets2[nn[1]])) / 2 if center else d)
    return res



def single_finder(ms, pix_sig=3, threshold=0.01, normalize=True):
    stack1 = ms.stack(0).data()[0,:,:,:]
    stack1 = np.array(stack1, np.float)
    return single_finder_inner(stack1, pix_sig, threshold, True, normalize)


def single_finder_inner(stack1, pix_sig, threshold, invertAxes, normalize):
    sig = pix_sig / np.sqrt(2)
    dets1 = detect_blobs(stack1, [sig, sig, sig], threshold, normalize)
    dets1 = list(dets1)
    if (len(dets1) == 0):
        return []
    kd1 = spatial.KDTree(dets1)
    cleanup_kdtree(stack1, kd1, dets1, 3)
    res = []
    for p in dets1:
        if invertAxes:
            res.append(list(p)[-1::-1])
        else:
            res.append(list(p))
    return res


def pair_finder(ms, pix_sig=3, thresholds=[0.01, 0.01], normalize=True):
    # get images in both channels
    stack1 = ms.stack(0).data()[0,:,:,:]
    stack2 = ms.stack(1).data()[0,:,:,:]
    stack1 = np.array(stack1, np.float)
    stack2 = np.array(stack2, np.float)

    return pair_finder_inner(stack1, stack2, pix_sig, thresholds, True, normalize)


def pair_finder_inner(stack1, stack2, pix_sig, threshold, invertAxes, normalize):
    # detect blobs via Laplacian-of-Gaussian (only blobs brighter than threshold)
    sig = pix_sig / np.sqrt(2)
    dets1 = detect_blobs(stack1, [sig, sig, sig], threshold[0], normalize)
    dets1 = list(dets1)
    dets2 = detect_blobs(stack2, [sig, sig, sig], threshold[0], normalize)
    dets2 = list(dets2)
    # did not find any spots in one of the channels -> return empty results
    if (len(dets1) == 0 or len(dets2) == 0):
        return []

    # put spots in kd-tree for fast nearesr neighbor calculations
    kd1 = spatial.KDTree(dets1)
    kd2 = spatial.KDTree(dets2)
    # if spots (in one channel) are closer than 3 pixels, pick the brighter of the two
    cleanup_kdtree(stack1, kd1, dets1, 3)
    cleanup_kdtree(stack2, kd2, dets2, 3)
    # for every remaining spot in image1, return a candidate pair if there is a spot in channel 2 that is closer than 5 pixels to it
    res = []
    for p in find_pairs(kd2, dets1, dets2, 5, invertAxes):
        res.append(list(p))
    return res


def detect_blobs(img, sigmas, threshold, normalize=False, threshold_rel_median=3, med_radius=5):
    img = skimage.util.img_as_float(img)

    # scale image to 0=min 1=max
    if normalize:
        img = (img - np.min(img)) / (np.max(img) - np.min(img))

    # do log filtering and local maxima detection
    logimg = -ndimage.gaussian_laplace(img, sigmas) * (sigmas[0] ** 2)
    peaks = skimage.feature.peak_local_max(logimg, threshold_abs=threshold)


    if threshold_rel_median:
        medimg = ndimage.median_filter(img, med_radius)
        peaks = [p for p in peaks if img[tuple(p)] > medimg[tuple(p)] * threshold_rel_median]

    return peaks    

