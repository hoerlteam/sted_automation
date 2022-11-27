from itertools import repeat

import numpy as np

def centered_tiles_1d(center_position, length, n_tiles, overlap):

    # integer steps
    x = np.arange(np.ceil(-n_tiles / 2), np.ceil(n_tiles / 2))
    # times distance considering overlap
    x *= length * (1 - overlap)

    # offset by half a tile of even number of tiles
    if not (n_tiles % 2):
        x += length * (1 - overlap) / 2

    x += center_position
    return x

def minmax_tiles_1d(min_position, max_position, length, overlap):

    # center of range
    center_position = (min_position + max_position) / 2

    # necessary number of tiles
    total_length = max_position - min_position
    n_tiles = np.ceil(total_length / (length * (1 - overlap))) + 1

    return centered_tiles_1d(center_position, length, n_tiles, overlap)

def alternate_axes(arr, alternate_along_axis, axes_to_alternate, offset=1):

    # wrap single axis to alternate
    if np.isscalar(axes_to_alternate):
        axes_to_alternate = (axes_to_alternate, )

    # copy arr as to not modify input
    arr=arr.copy()

    # select every second position along the axis to alternate on
    selection = tuple(slice(offset,s,2) if i==alternate_along_axis else slice(s) for i,s in enumerate(arr.shape))

    # perform inversion of the axes to alternate
    # note the clumsy slice(s, -(s+1), -1) seems to be necessary to represent [::-1] indexing
    alternation_selection = tuple(slice(s, -(s+1), -1) if i in axes_to_alternate else slice(s) for i,s in enumerate(arr[selection].shape))

    # perform actual alternation
    arr[selection] = arr[selection][alternation_selection]

    return arr

def centered_tiles(center_position, length, n_tiles, overlap, snake_rows=True):

    # repeat length, n_tiles, overlap if only scalar value is provided
    if np.isscalar(length):
        length = repeat(length)
    if np.isscalar(n_tiles):
        n_tiles = repeat(n_tiles)
    if np.isscalar(overlap):
        overlap = repeat(overlap)

    grid = np.meshgrid(*(centered_tiles_1d(x, l, n, o) for x, l, n, o in zip(center_position, length, n_tiles, overlap)), indexing='ij')
    grid = np.stack(grid, -1)

    if snake_rows :
        # alternate along rows
        grid = alternate_axes(grid, 0, 1)

    return grid.reshape((-1, len(center_position)))

def minmax_tiles(min_position, max_position, length, overlap, snake_rows=True):

    # repeat length, overlap if only scalar value is provided
    if np.isscalar(length):
        length = repeat(length)
    if np.isscalar(overlap):
        overlap = repeat(overlap)

    grid = np.meshgrid(*(minmax_tiles_1d(mi, ma, l, o) for mi, ma, l, o in zip(min_position, max_position, length, overlap)), indexing='ij')
    grid = np.stack(grid, -1)

    if snake_rows :
        # alternate along rows
        grid = alternate_axes(grid, 0, 1)

    return grid.reshape((-1, len(min_position)))