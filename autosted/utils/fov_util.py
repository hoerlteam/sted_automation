import numpy as np


def group_in_bounding_boxes(xs, bbox_size):
    """
    group points in bounding boxes of a given size.
    return the center (min+max/2) of each group.

    NB: this may not be the most efficient (minimal amount of bboxes) implementation.
    we basically find the closest point to origin and then cut bbox from there.
    this is repeated until all points are grouped.

    Parameters
    ----------
    xs: iterable of n-d points (array-likes)
        the points to group
    bbox_size:  n-d dimensions (array-like)
        dimension of the bounding boxes to group in

    Returns
    -------
    bbox_centers: iterable of n-d points (array-likes)
        center points of bounding boxes the xs were grouped in

    Raises
    ------
    ValueError: if dimensionality of any point != dimensionality of bbox
        or bbox_size == 0 in any dimension
    """

    worklist = [list(x) for x in xs]
    bbox_centers = []

    # check for incompatible arguments
    dims = len(bbox_size)
    if np.any(np.array(bbox_size) == 0):
        raise ValueError("cannot group in bounding boxes of zero extent")
    if np.any(np.array([len(x) for x in xs]) != dims):
        raise ValueError(
            "one or more of the given points does not match the dimensionality of bbox"
        )

    # while not empty
    while worklist:

        # all remaining points
        wl_tmp = [x for x in worklist]

        # find minimum for each dimension, discard points not in min+bbox for this dimension
        for d in range(len(wl_tmp[0])):
            min_d = np.min([x[d] for x in wl_tmp])
            wl_tmp = [x for x in wl_tmp if x[d] < min_d + bbox_size[d]]

        # remove points in the found bounding box from worklist
        for x in wl_tmp:
            worklist.remove(x)

        # get center of grouped points, add to result
        points_in_bbox = np.array(wl_tmp)
        center = (np.min(points_in_bbox, 0) + np.max(points_in_bbox, 0)) / 2.0
        bbox_centers.append(center)

    return bbox_centers
