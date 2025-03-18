from typing import Sequence
from itertools import product


class ResultsRepeater:
    """
    Simple callback to wrap another callback and repeat the results n times.

    E.g., can be used to image each location returned by a spot or ROI detector multiple times.
    """

    def __init__(
        self, wrapped_callback, n, make_nested=False, wrapped_callback_is_nested=False
    ):
        self.wrapped_callback = wrapped_callback
        self.n = n
        self.make_nested = make_nested
        self.wrapped_callback_is_nested = wrapped_callback_is_nested

    def __call__(self):
        results = self.wrapped_callback()
        repeated_values = []

        for values in results:

            # case 1: add multiple copies of values wrapped in a tuple,
            # this way, the nested values can become configurations in a wrapping building block
            if self.make_nested:
                # case 1a: results are already nested, just repaet
                if self.wrapped_callback_is_nested:
                    repeated_values.append(values * self.n)
                # case 1b: assume single results, we wrap in tuple and repeat
                else:
                    repeated_values.append((values,) * self.n)
            # case 2 (default): just add multiple copies, that is intended to result in multiple configurations
            else:
                for _ in range(self.n):
                    repeated_values.append(values)

        return repeated_values


class ValueCombinationsGenerator:
    """
    Callback that returns combinations (Cartesian product) of values from multiple value lists. 
    """

    def __init__(self, value_lists: Sequence[Sequence], n_repeats_per_combination=1, keep_looping=False):
        
        # check if list/sequence of lists
        if not (isinstance(value_lists, Sequence) and isinstance(value_lists[0], Sequence)):
            raise ValueError("value_lists should be list/sequence of lists of parameter values")

        self.value_lists = value_lists
        self.n_repeats_per_combination = n_repeats_per_combination
        self.keep_looping = keep_looping

        # build generator
        def _generator():
            while True:
                for value_combo in product(*self.value_lists):
                    for _ in range(self.n_repeats_per_combination):
                        yield value_combo
                if not self.keep_looping:
                    break

        # start generator
        self.generator = _generator()


    def get_all_combinations(self):

        """
        return all parameter values combinations at once, ignores keep_looping
        """

        results = []
        for value_combo in product(*self.value_lists):
            for _ in range(self.n_repeats_per_combination):
                results.append(value_combo)

        return results


    def __call__(self):

        # get next item from generator, None by default
        res = next(self.generator, None)

        # return empty result if no next item, else list containing that one item
        return [] if res is None else [res]
