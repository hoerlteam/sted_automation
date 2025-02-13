import logging

import numpy as np

from autosted.data import MeasurementData
from autosted.callback_buildingblocks.data_selection import NewestDataSelector


class AcceptanceCheck:

    def __init__(
        self,
        check_function,
        data_source_callback=None,
        configurations=(0,),
        channels=(0,),
        check_function_kwargs=None,
    ):
        
        """
        Parameters
        ----------
        check_function : a callable taking one or more images as the first positional arguments and optionally keyword arguments
            should return True ("image here") or False ("don't image here")
        data_source_callback : a callable (e.g. NewestDataSelector), which should return a MeasurementData object
        configurations : index of configuration to use, or list of indices to use multiple
        channels : index of channel to use, or list of indices to use multiple
        check_function_kwargs : keyword arguments to pass to check_function
        """

        if data_source_callback is None:
            data_source_callback = NewestDataSelector()
        self.data_source_callback = data_source_callback

        self.check_function = check_function

        # make sure we have a sequence of configurations & channels, even if just a single one is selected
        self.configurations = (
            (configurations,) if np.isscalar(configurations) else configurations
        )
        self.channels = (channels,) if np.isscalar(channels) else channels

        self.check_function_kwargs = (
            check_function_kwargs if check_function_kwargs is not None else {}
        )

        self.logger = logging.getLogger(__name__)

    def __call__(self):

        # get images of selected configurations, channels from data source
        measurement_data = self.data_source_callback()
        images = MeasurementData.collect_images_from_measurement_data(
            measurement_data, self.configurations, self.channels
        )

        # run the check function with images as arguments -> should return True for accept / False for reject
        check_result = self.check_function(*images, **self.check_function_kwargs)
        self.logger.info(
            "acceptance check result: {}".format("accept" if check_result else "reject")
        )

        # if we accept, we return a list of one dummy measurement with one configuration with empty update dicts
        if check_result:
            return [[({}, {})]]
        # if we reject, we return an empty list -> no measurements
        else:
            return []
