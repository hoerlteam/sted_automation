import logging
from itertools import cycle

import numpy as np

from pipeline2.utils.fov_util import group_in_bounding_boxes


class ParameterValuesRepeater:
    """
    Simple callback to wrap another callback and repeat the returned values n times.

    E.g., can be used to image each location returned by a spot or ROI detector multiple times.
    """

    def __init__(self, wrapped_callback, n=2, nested=False):
        self.wrapped_callback = wrapped_callback
        self.n = n
        self.nested = nested

    def __call__(self):
        value_sets = self.wrapped_callback()
        repeated_values = []

        for values in value_sets:

            # case 1: add multiple copies of values wrapped in a tuple,
            # this way, the nested values can become configurations in a wrapping building block
            if self.nested:
                repeated_values.append((values,) * self.n)
            # case 2 (default): just add multiple copies, that is intended to result in multiple configurations
            else:
                for _ in range(self.n):
                    repeated_values.append(values)

        return repeated_values


class SimpleManualOffset:

    """
    Callback to wrap another callback and add a manual offset to the returned parameter values.

    E.g., can be used to add a focus offset [z_offset, 0, 0] to results of spot detection
    """

    def __init__(self, wrapped_callback, offset):
        self.wrapped_callback = wrapped_callback
        self.offset = np.array(offset, dtype=float)

    def __call__(self):
        values = self.wrapped_callback()
        res = []

        offset_cycler = cycle(self.offset)
        for values_i in values:

            # we have a sequence of collection types (e.g., list of lists of coordinates)
            # add offset to all, keep structure
            if len(values_i) > 0 and not (np.isscalar(values_i[0]) or values_i[0] is None):
                ri_tup = []
                for values_i_inner in values_i:
                    ri = []
                    for value in values_i_inner:
                        off_i = next(offset_cycler)
                        ri.append(None if value is None else value + off_i)
                    ri_tup.append(ri)
                res.append(tuple(ri_tup))

            # we have just a list of value sequences (e.g., coordinate lists/arrays)
            else:
                ri = []
                for value in values_i:
                    off_i = next(offset_cycler)
                    ri.append(None if value is None else value + off_i)
                res.append(ri)
        return res


class BoundingBoxLocationGrouper:
    """
    Wrapper for a locationGenerator that groups locations into bounding boxes of defined size.
    This may be necessary to avoid multiple imaging of the same object.

    Parameters
    ----------
    location_generator : callable returning iterable of 3d location vectors
        generator of locations
    bounding_box_size : 3d vector (array-like)
        size of the bounding boxes to group in (same unit as vectors returned by locationGenerator)
    """
    def __init__(self, location_generator, bounding_box_size):
        self.location_generator = location_generator
        self.bounding_box_size = bounding_box_size
        self.logger = logging.getLogger(__name__)

    def __call__(self):
        xs = self.location_generator()
        res = group_in_bounding_boxes(xs, self.bounding_box_size)

        self.logger.info('grouped detections into {} FOVs'.format(len(res)))
        for loc in res:
            self.logger.debug('FOV: {}'.format(loc))
        return res


class LocalizationNumberFilter:
    """
    Wrapper for a locationGenerator that will discard all localizations, if there are too few or too many

    Parameters
    ----------
    location_generator : callable
        generator of locations
    min_num_locs: int, optional
        minimum number of localizations
    max_num_locs: int, optional
        maximum number of localizations
    """
    def __init__(self, location_generator, min_num_locs=None, max_num_locs=None):
        self.location_generator = location_generator
        self.min = min_num_locs
        self.max = max_num_locs

    def __call__(self):
        locs = self.location_generator.get_locations()
        n_locs = len(locs)

        # return all or nothing, depending on number of locs
        if self.min is not None and n_locs < self.min:
            return []
        elif self.max is not None and n_locs > self.max:
            return []
        else:
            return locs
