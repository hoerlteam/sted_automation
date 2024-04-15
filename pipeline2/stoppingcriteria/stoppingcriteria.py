from time import time

class TimedStoppingCriterion():
    """
    stopping criterion to stop after a set amount of time
    """

    def __init__(self, maxtime):
        self.maxtime = maxtime

    def check(self, pipeline):
        return time() > (pipeline.starting_time + self.maxtime)

    def desc(self, pipeline):
        return 'STOPPING PIPELINE {}: maximum time exceeded'.format(pipeline.name)


class InterruptedStoppingCriterion():
    """
    stopping criterion to check wether SIGINT was received and stop then
    will also reset the signal status in parent AcquisitionPipeline
    """

    def check(self, pipeline):
        return pipeline.interrupted

    def resetInterrupt(self, pipeline):
        pipeline.interrupted = False

    def desc(self, pipeline):
        return 'STOPPING PIPELINE {}: interrupted by user'.format(pipeline.name)