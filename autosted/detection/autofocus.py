import logging

import numpy as np
from scipy import ndimage as ndi

from autosted.utils.dict_utils import get_path_from_dict
from autosted.utils.parameter_constants import (
    OFFSET_STAGE_GLOBAL_PARAMETERS,
    PIXEL_SIZE_PARAMETERS,
)
from autosted.callback_buildingblocks.data_selection import NewestDataSelector


class SimpleFocusPlaneDetector:

    def __init__(
        self,
        data_source_callback=None,
        configuration=0,
        channel=0,
        invert_z_direction=True,
        focus_function=None,
        focus_function_kwargs=None,
    ):
        """
        Parameters
        ----------
        data_source_callback : an object implementing get_data(), which should return a MeasurementData object
        configuration : int, index of configuration to use for focus
        channel : int, index of channel to use for focus
        verbose : bool, set to True for extra debugging output
        invert_z_direction: whether to invert the z direction of the focus update
            if the update should change stage focus, but the image stack was acquired using the piezo drive this should be True
        focus_function : a function that takes an image and returns a focus delta
            if none is given, by default, will detect the plane with the highest mean intensity
        focus_function_kwargs : a dictionary of keyword arguments to pass to the focus function
        """

        if data_source_callback is None:
            data_source_callback = NewestDataSelector()
        self.data_source_callback = data_source_callback

        self.configuration = configuration
        self.channel = channel
        self.logger = logging.getLogger(__name__)
        self.invert_z_direction = invert_z_direction

        if focus_function is None:
            self.focus_function = SimpleFocusPlaneDetector.mean_intensity_focus
        self.focus_function_kwargs = (
            focus_function_kwargs if focus_function_kwargs is not None else {}
        )

        self.offset_z_path = OFFSET_STAGE_GLOBAL_PARAMETERS[0]
        self.pixel_size_z_path = PIXEL_SIZE_PARAMETERS[0]

    @staticmethod
    def mean_along_axis(img, axis):
        """
        calculate mean of every hyper-slice in image along axis
        """
        axes = tuple([ax for ax in range(len(img.shape)) if ax != axis])
        profile = np.mean(img, axes)
        return profile

    @staticmethod
    def mean_intensity_focus(img, pixel_size, axis=0, sigma=3):
        # TODO: move pixel size out of the function
        # get mean profile, smooth it via a Gaussian blur
        profile = SimpleFocusPlaneDetector.mean_along_axis(img, axis)
        smooth_profile = ndi.gaussian_filter1d(profile, sigma=sigma, mode="constant")
        profile_max = np.argmax(smooth_profile)
        # calculate offset of maximum in comparison to middle
        pix_d = profile_max - ((len(profile) - 1) / 2)
        return pix_d * pixel_size

    def __call__(self):

        data = self.data_source_callback()

        # no data yet -> empty update
        if data is None:
            self.logger.info(": No data for Z correction present -> skipping.")
            return [[None, None, None]]

        if (data.num_configurations <= self.configuration) or (
            data.num_channels(self.configuration) <= self.channel
        ):
            raise ValueError("no images present. TODO: fail gracefully/skip here")

        # get image of selected configuration and channel and convert to float
        img = data.data[self.configuration][self.channel][0, :, :, :]
        img = np.array(img, float)

        # 2D image -> empty update
        if img.shape[0] <= 1:
            self.logger.info(": Image is 2D, cannot do Z correction -> skipping.")
            return [[None, None, None]]

        # get old z-offset and pixel size
        setts = data.measurement_settings[self.configuration]
        z_offset_old = get_path_from_dict(
            setts, self.offset_z_path, keep_structure=False
        )
        z_pixel_size = get_path_from_dict(
            setts, self.pixel_size_z_path, keep_structure=False
        )

        # get z delta, add or subtract from old z-offset
        z_delta = self.focus_function(
            img, z_pixel_size, 0, **self.focus_function_kwargs
        )
        new_z = z_offset_old + z_delta * (-1 if self.invert_z_direction else 1)

        self.logger.info(
            ": Corrected Focus (was {}, new {})".format(z_offset_old, new_z)
        )

        return [[new_z, None, None]]
