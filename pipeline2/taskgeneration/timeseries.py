import warnings
from time import sleep, time


class DummyAcquisitionTask:

    """
    A dummy acquisition task with no measurements / configurations that can be added to the queue to trigger callbacks.
    """

    def __init__(self, pipeline_level) -> None:
        self.pipeline_level = pipeline_level
        # setting this to 0 will prevent Pipeline from querying for settings
        self.num_acquisitions = 0

    def __len__(self):
        return 0


class TimeSeriesCallback:

    def __init__(self, pipeline_level, time_points=(0.0,)) -> None:
        self.pipeline_level = pipeline_level
        self.re_initialized = True
        self.start_time = 0
        self.time_points = time_points
        self.current_time_point_idx = 0

        # seconds to wait before warning about overtime of previous acquisition
        self.max_wait_before_warn = 5

    def __call__(self):

        # reset timer and pointer to current time point
        if self.re_initialized:
            self.re_initialized = False
            self.start_time = time()
            self.current_time_point_idx = 0

        # catch edge cases
        # normally, no new TimeSeriesDummyAcquisitionTasks should be enqueued after last time point (see below)
        if self.current_time_point_idx >= len(self.time_points):
            warnings.warn('time series callback called on already finished time series.')
            return

        next_tp = self.time_points[self.current_time_point_idx]

        wait_time = next_tp - (time() - self.start_time)
        if wait_time < (- self.max_wait_before_warn):
            warnings.warn(f'Next time point in time series was scheduled {-wait_time} seconds ago, but previous acquisition(s) did not finish in time.')

        # wait until next time point is due
        sleep(max(0.0, wait_time))

        # increment time point index
        self.current_time_point_idx += 1

        # only enqueue dummy acquisition task if there is a next time point
        if self.current_time_point_idx <= len(self.time_points):
            return self.pipeline_level, DummyAcquisitionTask(self.pipeline_level)