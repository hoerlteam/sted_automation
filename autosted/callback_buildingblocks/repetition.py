class ResultsRepeater:
    """
    Simple callback to wrap another callback and repeat the results n times.

    E.g., can be used to image each location returned by a spot or ROI detector multiple times.
    """

    def __init__(self, wrapped_callback, n, nested=False):
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