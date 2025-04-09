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
            # as Python dicts are ordered, last key should be newest
            newest_data_idx = list(self.pipeline.data.keys())[-1]
            # index length to corresponding level
            self.level = newest_data_idx[-1][0]

        # get all other indices of same level
        indices_same_level = [
            (lvl, idx) for (lvl, idx) in self.pipeline.data.keys() if lvl == self.level
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
