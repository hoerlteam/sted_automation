from autosted import AcquisitionPipeline
from autosted.data import MeasurementData


class NewestDataSelector:
    """
    Callback that will return the newest MeasurementData at a given hierarchy level from the pipeline
    """

    def __init__(self, pipeline=None, level=None):
        self.pipeline: AcquisitionPipeline = pipeline
        self.level = level

    def __call__(self):

        # no pipeline reference was given
        # -> use currently running instance at first callback usage
        if self.pipeline is None:
            if AcquisitionPipeline.running_instance is None:
                raise ValueError("No running AcquisitionPipeline found")
            self.pipeline = AcquisitionPipeline.running_instance

        # no level was given
        # -> use newest data level at first callback usage
        if self.level is None:
            # no data yet, postpone
            if len(self.pipeline.data) == 0:
                return None
            # sort indices -> largest one should be newest
            newest_data_idx = sorted(self.pipeline.data.keys())[-1]
            # index length to corresponding level
            self.level = self.pipeline.hierarchy_levels[len(newest_data_idx) - 1]

        # length of indices of same level: position in levels of pipeline + 1
        index_length = self.pipeline.hierarchy_levels.index(self.level) + 1
        # get all measurement indices in data of same length (same level)
        indices_same_level = [
            k for k in self.pipeline.data.keys() if len(k) == index_length
        ]

        # if no data, return None
        if len(indices_same_level) == 0:
            return None

        # as index tuples are increasing, latest measurement will be first in reverse-sorted indices
        latest_index = sorted(indices_same_level, reverse=True)[0]
        return self.pipeline.data.get(latest_index, None)


class NewestSettingsSelector(NewestDataSelector):
    """
    Callback that will return just the JSON-settings of the newest MeasurementData
    at a given hierarchy level from the pipeline
    """

    def __call__(self):
        data: MeasurementData = super().__call__()
        if data is None:
            return None
        return [list(zip(data.measurement_settings, data.hardware_settings))]
