import logging
import numpy as np
from matplotlib import pyplot as plt

from autosted.utils.coordinate_utils import virtual_bbox_from_settings, PIXEL_SIZE_PARAMETERS, get_parameter_value_array_from_dict
from autosted.utils.coordinate_utils import FOV_LENGTH_PARAMETERS, OFFSET_STAGE_GLOBAL_PARAMETERS, OFFSET_SCAN_PARAMETERS, OFFSET_SCAN_GLOBAL_PARAMETERS
from autosted.utils.dict_utils import get_path_from_dict
from autosted.callback_buildingblocks import ScanModeSettingsGenerator
from autosted.data import HDF5DataReader

from calmutils.stitching import stitch
from calmutils.stitching.fusion import fuse_image
from calmutils.stitching.transform_helpers import translation_matrix
from calmutils.stitching.transform_helpers import scale_matrix
from calmutils.imageio import save_tiff_imagej


class DemoImspectorConnection:

    def __init__(self, path_h5_dataset, plot_images=True):

        # current ("acquired") data
        self.measurement_parameters = []
        self.data = []

        self.plot_images = plot_images

        dataset = HDF5DataReader(path_h5_dataset)

        self.imgs = []
        self.transforms = []
        self.pixel_sizes = []

        for measurement in dataset:
            measurement_settings = dataset[measurement].measurement_settings[0]
            start, fov = virtual_bbox_from_settings(measurement_settings)
            pixel_size = get_parameter_value_array_from_dict(measurement_settings, PIXEL_SIZE_PARAMETERS)

            # build (pixel-unit) transform matrix from coordinates
            transform = translation_matrix((start / pixel_size))
            # get data
            img = dataset[measurement].data[0][0].squeeze()

            self.imgs.append(img)
            self.transforms.append(transform)
            self.pixel_sizes.append(pixel_size)

        # stitch - will give transforms relative to first img
        reference_tr = self.transforms[0]
        self.transforms = stitch(self.imgs, [tr[:-1,-1] for tr in self.transforms])
        # re-apply first transform
        self.transforms = [tr @ reference_tr for tr in self.transforms]

        self.logger = logging.getLogger(__name__)

    def make_measurement_from_task(self, task):
        # append measurement parameters to parameter list
        # NOTE: hardware parameters are ignored in simulation
        measurement_updates, hardware_updates = task
        self.measurement_parameters.append(measurement_updates)
        pass

    def make_configuration_from_task(self, task):
        self.make_measurement_from_task(task)

    def run_current_measurement(self):

        # 1. get last parameters
        parameters = self.measurement_parameters[-1]

        # check scan mode -> we only support xy, xyz
        scan_mode = get_path_from_dict(parameters, ScanModeSettingsGenerator._path_scanmode, False)
        if scan_mode not in ("xy", "xyz"):
            raise ValueError("Only XY and XYZ scan supported in simulated acquisitions")

        # 2. get FOV / pixel size described by them
        start, fov = virtual_bbox_from_settings(parameters)
        pixel_size = get_parameter_value_array_from_dict(parameters, PIXEL_SIZE_PARAMETERS)


        # get acquisition coordinates / fov and log for debug
        params = parameters
        offsets_stage = [
            get_path_from_dict(params, path, False)
            for path in OFFSET_STAGE_GLOBAL_PARAMETERS
        ]
        offsets_scan = [
            get_path_from_dict(params, path, False) for path in OFFSET_SCAN_PARAMETERS
        ]
        fov_length = [
            get_path_from_dict(params, path, False) for path in FOV_LENGTH_PARAMETERS
        ]

        self.logger.info("running simulated acquisition:")
        self.logger.info("stage offsets: {}".format(offsets_stage))
        self.logger.info("scan offsets: {}".format(offsets_scan))
        self.logger.info("scan mode: {}".format(scan_mode))
        self.logger.info("FOV length: {}".format(fov_length))

        # 3. fuse virtual image

        bbox_start = np.round(start / pixel_size).astype(int)
        bbox_end = bbox_start + np.round(fov / pixel_size).astype(int)

        if scan_mode == "xy":
            bbox_start[0] = ((bbox_start[0] + bbox_end[0]) / 2).astype(int)
            bbox_end[0] = bbox_start[0] + 1

        bbox = list(zip(bbox_start, bbox_end))

        transforms_scaled = []
        for (tr, pixel_size_ref) in zip (self.transforms, self.pixel_sizes):
            scale_factors = pixel_size_ref / pixel_size
            transforms_scaled.append(scale_matrix(scale_factors) @ tr)

        fused_img = fuse_image(self.imgs, transforms_scaled, bbox)

        # TODO: plot image
        if self.plot_images:
            plt.imshow(fused_img.max(0), cmap="gray")
            plt.show()
        
        # add dummy dimesions so we have 4D stack as in Imspector
        dummy_dims = (1,) * (4 - fused_img.ndim)
        stack = fused_img.reshape(dummy_dims + fused_img.shape)

        self.data.append([stack])

    def get_current_data(self):
        return {}, self.measurement_parameters[-1], self.data[-1]

    def save_current_measurement(self, path):
        # get path without ending
        path = path.rsplit(".", 1)[-2] + ".tif"
        save_tiff_imagej(path, self.data[-1][0], "TZYX")

    def close_current_measurement(self):
        # reset parameters / data lists
        self.measurement_parameters = []
        self.data = []