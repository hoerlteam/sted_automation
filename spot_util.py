from skimage.feature import blob_dog, blob_log, blob_doh
from scipy import ndimage, spatial, stats
import numpy as np
import skimage

def mean_along_axis(img, axis):
    axes = tuple([ax for ax in range(len(img.shape)) if ax != axis])
    #print(axes)
    profile = np.mean(img, axes)
    return profile

def focus_in_stack(img, pixsize_z):
    profile = mean_along_axis(img, 1)
    smoothprofile = ndimage.gaussian_filter1d(profile, 3, mode='constant')
    tmax = np.argmax(smoothprofile)
    pix_d = tmax - (len(profile) / 2)
    return pix_d * pixsize_z

def cleanup_kdtree(img, kdt, dets, dist):
    notdone = True
    while notdone:
        notdone = False
        for i in range(len(dets)):
            nn = kdt.query(dets[i],k=2, distance_upper_bound=dist)
            if (not np.isinf(nn[0][1])) and nn[0][1] > 0:
                ineighbor = nn[1][1]
                if img[tuple(dets[i])] < img[tuple(dets[ineighbor])]:
                    dets.pop(i)
                else:
                    dets.pop(ineighbor)
                notdone = True
                    
                kdt = spatial.KDTree(dets)
                break
        
def find_pairs(kdt2, dets1, dist):
    res = []
    for d in dets1:
        nn = kdt2.query(d, distance_upper_bound=dist)
        if not np.isinf(nn[0]):
            # dets were in zyx -> turn to xyz
            res.append(d[-1::-1])
    return res

def single_finder(ms, pix_sig=3, threshold=0.01):
    stack1 = ms.stack(0).data()[0,:,:,:]
    sig = pix_sig / np.sqrt(2)
    dets1 = detect_blobs(stack1, [sig, sig, sig], threshold)    
    dets1=list(dets1)
    
    if (len(dets1) == 0):
        return []
    
    kd1 = spatial.KDTree(dets1)
    cleanup_kdtree(stack1, kd1, dets1, 3)
    
    res = []
    for p in dets1:
        res.append(list(p)[-1::-1])
    return res

def pair_finder(ms, pix_sig=3, threshold=0.01):

    stack1 = ms.stack(0).data()[0,:,:,:]
    stack2 = ms.stack(1).data()[0,:,:,:]

    sig = pix_sig / np.sqrt(2)
    dets1 = detect_blobs(stack1, [sig, sig, sig], threshold)    
    dets1=list(dets1)
    dets2 = detect_blobs(stack2, [sig, sig, sig], threshold)    
    dets2=list(dets2)

    if (len(dets1) == 0 or len(dets2) == 0):
        return []
    
    kd1 = spatial.KDTree(dets1)
    kd2 = spatial.KDTree(dets2)

    cleanup_kdtree(stack1, kd1, dets1, 3)
    cleanup_kdtree(stack2, kd2, dets2, 3)

    res = []
    for p in find_pairs(kd2, dets1, 5):
        res.append(list(p))

    return res

def detect_blobs(img, sigmas, threshold):
    img = skimage.util.img_as_float(img)
    img = img / np.max(img)
    logimg = -ndimage.gaussian_laplace(img, sigmas) #* (sigmas[0] ** 2)
    peaks = skimage.feature.peak_local_max(logimg, threshold_abs=threshold)
    return peaks    

