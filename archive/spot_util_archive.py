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


def pair_finder_yellow(ms, pix_sig=3, thresholds=[0.01, 0.01], normalize=True, median_thresholds=[3, 3],
                       median_radius=5):
    stack1 = ms.stack(0).data()[0, :, :, :]
    stack2 = ms.stack(1).data()[0, :, :, :]
    stack1 = np.array(stack1, np.float)
    stack2 = np.array(stack2, np.float)
    return pair_finder_yellow_inner(stack1, stack2, pix_sig, thresholds, True, normalize, median_thresholds,
                                    median_radius)


def pair_finder(ms, pix_sig=3, thresholds=[0.01, 0.01], normalize=True, median_thresholds=[3, 3], median_radius=5):
    # get images in both channels
    stack1 = ms.stack(0).data()[0, :, :, :]
    stack2 = ms.stack(1).data()[0, :, :, :]
    stack1 = np.array(stack1, np.float)
    stack2 = np.array(stack2, np.float)

    return pair_finder_inner(stack1, stack2, pix_sig, thresholds, True, normalize, median_thresholds, median_radius)


def pair_finder_yellow_inner(stack1, stack2, pix_sig, threshold, invertAxes, normalize, median_thresholds,
                             median_radius):
    # detect blobs via Laplacian-of-Gaussian (only blobs brighter than threshold)
    sig = pix_sig / np.sqrt(2)
    dets1 = detect_blobs(stack1, [sig, sig, sig], threshold[0], normalize, median_thresholds[0], median_radius)
    dets1 = list(dets1)
    dets2 = detect_blobs(stack2, [sig, sig, sig], threshold[1], normalize, median_thresholds[1], median_radius)
    dets2 = list(dets2)

    dets = dets1 + dets2
    # did not find any spots in any of the channels -> return empty results
    if (len(dets) == 0):
        return []

    # put spots in kd-tree for fast nearesr neighbor calculations
    kd = spatial.KDTree(dets)
    # if spots (in any channel) are closer than 3 pixels, pick the brighter of the two
    # quick hack: we take the brighter spot in averaged channels
    cleanup_kdtree((stack1 + stack2) / 2, kd, dets, 3)
    res = []
    for p in dets:
        if invertAxes:
            res.append(list(p)[-1::-1])
        else:
            res.append(list(p))
    return res