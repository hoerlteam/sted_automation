import numpy as np
import fastremap


def greedy_mesh(mask, threshold=10, relaxed=True):

    """
    Row-wise greedy meshing of mask into rectangles have size <= threshold in at least one dimension.
    Relaxed means we extend a rectangle to the next row even if only some but not all pixels are labelled.
    """

    # ensure we have boolean mask
    mask = mask.astype(bool)

    # allow setting common threshold or height, width threshold separately
    if np.isscalar(threshold):
        threshold_h, threshold_w = threshold, threshold
    else:
        threshold_h, threshold_w = threshold

    final_rects = []

    # active rects: set of (y_start, x_start, current_height, width) tuples
    active_rects = set()

    for row_idx, row in enumerate(mask):

        # new set of active rects for the next iteration
        next_active_rects = set()

        # which pixels in this row are claimed by extending existing rects
        claimed_mask = np.zeros(row.shape, dtype=bool)

        # --- PHASE 1: Try to extend existing rectangles ---
        for (y_start, x_start, current_height, width) in active_rects:

            # The footprint of the rect on the current row
            x_end = x_start + width
            footprint = row[x_start : x_end]

            # We continue if the footprint contains ANY useful data (1s).
            contains_structure = np.any(footprint) if relaxed else np.all(footprint)

            # Check size limits
            new_height = current_height + 1
            is_valid_size = (width <= threshold_w) or (new_height <= threshold_h)
            
            if contains_structure and is_valid_size:
                next_active_rects.add((y_start, x_start, new_height, width))
                # Mark these pixels as claimed so Phase 2 doesn't try to mesh them
                claimed_mask[x_start : x_end] = True
            else:
                # Seal it
                final_rects.append((y_start, x_start, current_height, width))
        
        # --- PHASE 2: Find new strips in the remaining gaps ---

        # 1s that are NOT in claimed_mask (pad so we can find border at start and end)
        remaining_row = np.pad((row & (~ claimed_mask)), 1)
        borders = np.diff(remaining_row)

        # borders will be 1 at starts and ends of runs, go over in batches of 2
        border_idxs = np.flatnonzero(borders)
        for i in range(0, len(border_idxs), 2):
            start, end = border_idxs[i:i+2]
            next_active_rects.add((row_idx, start.item(), 1, (end-start).item()))

        # active rects for next iteration
        active_rects = next_active_rects

    # end of vertical pass: seal leftovers
    final_rects += active_rects

    return final_rects


def greedy_mesh_3d(mask, threshold=10, relaxed=True):

    """
    Extension of greedy_mesh to 3D.
    Will process plane by plane, extending rectalngles from previous plane to form cuboids.
    For areas that can not be extended, we call the 2D greedy_mesh to start new candidates.

    TODO: merge with 2D case, logic is pretty identical?
    """

    # ensure we have boolean mask
    mask = mask.astype(bool)

    # allow setting common threshold or depth, height, width threshold separately
    if np.isscalar(threshold):
        threshold_d, threshold_h, threshold_w = threshold, threshold, threshold
    else:
        threshold_d, threshold_h, threshold_w = threshold

    final_cuboids = []

    # active cuboids: (z, y, x, d, h, w)
    active_cuboids = set()

    for z_idx, plane in enumerate(mask):

        # new set of active rects for the next iteration
        next_active_cuboids = set()

        # which pixels in this plane are claimed by extending existing cuboids
        claimed_mask = np.zeros(plane.shape, dtype=bool)

        # --- PHASE 1: Try to extend existing cuboids ---
        for (z, y, x, d, h, w) in active_cuboids:

            # The footprint of the rect on the current slice
            footprint = plane[y:y+h, x:x+w]

            # We continue if the footprint contains ANY useful data (1s).
            contains_structure = np.any(footprint) if relaxed else np.all(footprint)

            # Check size limits
            # we sum checks -> at least 2 of 3 have to hold
            new_depth = d + 1
            is_valid_size = sum([new_depth <= threshold_d, w <= threshold_w, h <= threshold_h]) >= 2
            
            if contains_structure and is_valid_size:
                next_active_cuboids.add((z, y, x, new_depth, h, w))
                # Mark these pixels as claimed so Phase 2 doesn't try to mesh them
                claimed_mask[y:y+h, x:x+w] = True
            else:
                # Seal it
                final_cuboids.append((z, y, x, d, h, w))

        # 1s that are NOT in claimed_mask (pad so we can find border at start and end)
        remaining_plane = (plane & (~ claimed_mask))

        # greedy mesh remaining mask in 2d, open new active cubes
        for (y, x, h, w) in greedy_mesh(remaining_plane, (threshold_h, threshold_w), relaxed):
            next_active_cuboids.add((z_idx, y, x, 1, h, w))

        # active rects for next iteration
        active_cuboids = next_active_cuboids

    # end of depthwise pass: seal leftovers
    final_cuboids += active_cuboids

    return final_cuboids


def get_tiles(shape, tile_size=10, offset=0):

    """
    Get a labelled array of given shape covered in tiles of given size with unique ids
    """

    ndim = len(shape)

    # we allow unequal tile sizes in different dims
    # if only one size is given, repeat
    if np.isscalar(tile_size):
        tile_size = (tile_size, ) * ndim

    # same for offset
    if np.isscalar(offset):
        offset = (offset, ) * ndim

    # offset only has affect mod tile_size
    offset = tuple(off % tile_size_i for (off, tile_size_i) in zip(offset, tile_size))

    # how many tiles do we need (shape/tilesize + 1), extra +1 because we have up to tilesize extra offset
    n_tiles_shape = tuple(int(s/tile_size_i) + 2 for s, tile_size_i in zip(shape, tile_size))
    n_tiles = np.prod(n_tiles_shape)

    # build non-offset tiles
    tiles_full = np.arange(1, n_tiles+1).reshape(n_tiles_shape)
    for d, n in enumerate(n_tiles_shape):
        tiles_full = np.repeat(tiles_full, tile_size[d], d)

    # roll for offset
    for d, off in enumerate(offset):
        tiles_full = np.roll(tiles_full, off, d)

    # result is bigger than shape, cut to shape
    tiles_full = tiles_full[*(slice(0,s) for s in shape)]

    # renumber so we start at 1 in the top/left
    tiles_full, _ = fastremap.renumber(tiles_full, in_place=True)

    return tiles_full


def get_tiled_mask(mask, tile_size=10, offset=0):

    """
    get label image of tiles (see get_tiles) that cover a boolean mask.
    """

    # get tiles
    tiles = get_tiles(mask.shape, tile_size, offset)

    # ensure bool mask (0, 1)
    mask = mask.astype(bool)

    # all tile idxs
    all_idxs = fastremap.unique(tiles)
    # multiplying tiles with mask sets tiles to 0 where mask is absent
    present_idxs = set(fastremap.unique(tiles * mask))

    # prepare remap dict to keep idx if present else map to 0
    remap_dict = {idx: idx if idx in present_idxs else 0 for idx in all_idxs}

    # remap and renumber
    tiles = fastremap.remap(tiles, remap_dict, in_place=True)
    tiles, _ = fastremap.renumber(tiles, in_place=True)

    return tiles


def get_tiled_mask_optimal(mask, tile_size=10, return_offset=True, step=1):

    """
    Get label image of tiles that cover mask.
    Will try all possible offsets in 0 .. tile_size and return the tiling that has the highest mean pixels per tile.
    If step > 1, will only try offsets that are multiples of step
    """

    # ensure bool mask (0, 1)
    mask = mask.astype(bool)

    # we allow unequal tile sizes in different dims
    # if only one size is given, repeat
    if np.isscalar(tile_size):
        tile_size = (tile_size, ) * mask.ndim

    # list of all possible offsets
    offsets = np.mgrid[*(slice(0, si, step) for si in tile_size)].reshape((len(tile_size), -1)).T

    optimal_mask = None
    optimal_offset = None
    optimal_pixels_per_tile = 0

    for offset in offsets:

        # get mask with offset, calculate mean pixels per tiles
        tiled_mask = get_tiled_mask(mask, tile_size, offset)
        values, counts = fastremap.unique(tiled_mask * mask, return_counts=True)
        mean_pixels_per_tile = counts[1:].sum() / (len(values) - 1)

        # keep if we have highest average pixels per tile
        if mean_pixels_per_tile > optimal_pixels_per_tile:
            optimal_pixels_per_tile = mean_pixels_per_tile
            optimal_mask = tiled_mask
            optimal_offset = offset

    return (optimal_mask, optimal_offset) if return_offset else optimal_mask