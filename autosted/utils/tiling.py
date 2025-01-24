# import more generic functions that were moved to CalmUtils
from calmutils.stitching.tiling import centered_tiles, minmax_tiles


def relative_spiral_generator(steps, start=[0, 0]):
    """
    generator for two-dimensional regular spiral coordinates around a starting point
    with given step sizes
    """

    # single tile in center
    yield start[0:2].copy()

    n = 1
    while True:
        # move n rows "left & down"
        bookmark = [-n * steps[0] + start[0], n * steps[1] + start[1]]
        for _ in range(2 * n):
            yield bookmark.copy()
            bookmark[0] += steps[0]
        for _ in range(2 * n):
            yield bookmark.copy()
            bookmark[1] -= steps[1]
        for _ in range(2 * n):
            yield bookmark.copy()
            bookmark[0] -= steps[0]
        for _ in range(2 * n):
            yield bookmark.copy()
            bookmark[1] += steps[1]
        n += 1
