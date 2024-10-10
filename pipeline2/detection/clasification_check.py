import logging
from pipeline2.data import MeasurementData


class AcceptanceCheck:

    def __init__(self, data_source_callback, check_function, configurations=(0,), channels=(0,),
                 check_function_kwargs=None):

        self.data_source_callback = data_source_callback
        self.check_function = check_function

        # make sure we have a sequence of configurations & channels, even if just a single one is selected
        self.configurations = (configurations,) if np.isscalar(configurations) else configurations
        self.channels = (channels,) if np.isscalar(channels) else channels

        self.check_function_kwargs = check_function_kwargs if check_function_kwargs is not None else {}

        self.logger = logging.getLogger(__name__)

    def __call__(self):

        # get images of selected configurations, channels from data source
        measurement_data = self.data_source_callback()
        images = MeasurementData.collect_images_from_measurement_data(measurement_data, self.configurations, self.channels)

        # run the check function with images as arguments -> should return True for accept / False for reject
        check_result = self.check_function(*images, **self.check_function_kwargs)
        self.logger.info('acceptance check result: {}'.format('accept' if check_result else 'reject'))

        # if we accept, we return a list of one dummy measurement with one configuration with empty update dicts
        if check_result:
            return [[({}, {})]]
        # if we reject, we return an empty list -> no measurements
        else:
            return []


if __name__ == '__main__':

    import numpy as np
    from pipeline2.taskgeneration import AcquisitionTaskGenerator
    from pipeline2.callback_buildingblocks.static_settings import ScanModeSettingsGenerator
    from pipeline2.data import MeasurementData

    logging.basicConfig(level=logging.INFO)

    data = MeasurementData()
    data.append({}, {}, np.zeros((1,1,100,100)))
    data_call = lambda: data

    gen = AcquisitionTaskGenerator('test', ScanModeSettingsGenerator('xy'), AcceptanceCheck(data_call, lambda *x: True))
    _, task = gen()
    print(task[0].get_all_updates(True))