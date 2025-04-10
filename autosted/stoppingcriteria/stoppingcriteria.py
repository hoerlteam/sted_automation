from time import time


class MaximumAcquisitionsStoppingCriterion:

    def __init__(self, max_acquisitions=None, max_acquisitions_per_level=None):
        self.max_acquisitions = max_acquisitions
        self.max_acquisitions_per_level = max_acquisitions_per_level

    def check(self, pipeline):

        # we have set no limits -> always return False
        if self.max_acquisitions is None and self.max_acquisitions_per_level is None:
            return False

        # check total amount of acquisitions
        total_acquisitions = len(pipeline.data)
        if (
            self.max_acquisitions is not None
            and total_acquisitions >= self.max_acquisitions
        ):
            return True

        # check acquisitions per level
        if self.max_acquisitions_per_level is not None:
            for (
                level,
                max_acquisitions_at_level,
            ) in self.max_acquisitions_per_level.items():

                # get number of all measurement indices in data of same length (same level)
                num_acquisitions_level = len(
                    [k for k in pipeline.data.keys() if k[-1][0] == level]
                )
                if num_acquisitions_level >= max_acquisitions_at_level:
                    return True

        return False

    def desc(self, pipeline):
        return "STOPPING PIPELINE {}: maximum number of acquisitions reached".format(
            pipeline.name
        )


class TimedStoppingCriterion:
    """
    stopping criterion to stop after a set amount of time (given in seconds)
    """

    def __init__(self, max_time_sec):
        self.max_time_sec = max_time_sec

    def check(self, pipeline):
        return time() > (pipeline.starting_time + self.max_time_sec)

    def desc(self, pipeline):
        return "STOPPING PIPELINE {}: maximum time exceeded".format(pipeline.name)


class InterruptedStoppingCriterion:
    """
    stopping criterion to check whether SIGINT was received and stop then
    will also reset the signal status in parent AcquisitionPipeline
    """

    def check(self, pipeline):
        if not pipeline.interrupted:
            return False
        else:
            # reset the interrupt in pipeline (e.g. so it can be run again)
            # TODO: check if really necessary?
            self.reset_interrupt(pipeline)
            return True

    def reset_interrupt(self, pipeline):
        pipeline.interrupted = False

    def desc(self, pipeline):
        return "STOPPING PIPELINE {}: interrupted by user".format(pipeline.name)
