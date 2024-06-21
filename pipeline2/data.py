import h5py
import json
import re
from collections import defaultdict
import numpy as np


class MeasurementData:
    """
    wrapper for data and settings/parameters of a measurement
    holds hardware settings and measurement settings
    for each parameter set (configuration) and a list of associated data
    """

    def __init__(self):
        self.hardware_settings = []
        self.measurement_settings = []
        self.data = []

    # TODO: remove defaults?
    def append(self, hardware_settings=None, measurement_settings=None, data=None):
        self.hardware_settings.append(hardware_settings)
        self.measurement_settings.append(measurement_settings)
        self.data.append(data)

    @property
    def num_configurations(self):
        return len(self.data)

    def num_channels(self, configuration_idx):
        if configuration_idx < self.num_configurations:
            return len(self.data[configuration_idx])
        else:
            return 0


class HDF5DataStore(defaultdict):

    def __init__(self, h5_file, pipeline_levels=None, root_path='experiment', read_only=False, compressed=True):
        super().__init__()
        self.h5_file = h5_file

        self.read_only = read_only
        file_mode = 'r' if read_only else 'a'

        self.root_path = root_path
        self.pipeline_levels = pipeline_levels

        self.compressed = compressed

        # get file handle for write: if member h5_file is already a File object, use as-is
        # otherwise, we assume h5_file is a str/Path and try to open in append mode
        file_is_file_handle = isinstance(self.h5_file, h5py.File)
        fd = self.h5_file if file_is_file_handle else h5py.File(self.h5_file, file_mode)

        # add root group and information about pipeline levels to h5 file
        if root_path not in fd:
            if self.read_only:
                raise ValueError('trying to create data store from existing file, but file is empty')
            else:
                fd.create_group(root_path)
        attrs = h5py.AttributeManager(fd[root_path])

        if 'levels' in attrs:
            levels_from_file = attrs['levels'].split(',')
        else:
            levels_from_file = None

        if self.pipeline_levels is None:
            self.pipeline_levels = levels_from_file

        if not self.read_only:
            if levels_from_file is not None:
                # TODO: clean warning/logging
                print('WARNING: overwriting existing levels')
            attrs['levels'] = ','.join(self.pipeline_levels)

        # if we have opened a new file object, close it again
        # otherwise (we are using a provided object), leave it
        if not file_is_file_handle:
            fd.close()

        self.load_existing_data()

    def load_existing_data(self):
        for idx in self._get_indices():
            self[idx]

    def _get_indices(self):
        """
        get all the numerical indices of data in the hdf5 file
        :return: list if index tuples
        """

        p = re.compile('(\\d*)(?:_){,1}'.join(map(lambda l : '(?:{}){{,1}}'.format(l), self.pipeline_levels)) + '(\\d*)')
        idxes = []

        # get file handle for write: if member h5_file is already a File object, use as-is
        # otherwise, we assume h5_file is a str/Path and try to open in append mode
        file_is_file_handle = isinstance(self.h5_file, h5py.File)
        fd = self.h5_file if file_is_file_handle else h5py.File(self.h5_file, 'r')

        for k in fd[self.root_path].keys():
            if not p.match(k):
                continue
            g = p.match(k).groups()
            g = tuple([int(gi) for gi in g if gi != ''])
            idxes.append(g)

        # if we have opened a new file object, close it again
        # otherwise (we are using a provided object), leave it
        if not file_is_file_handle:
            fd.close()

        return idxes

    def __missing__(self, key):

        new_data = HDF5MeasurementData(self.h5_file, self.pipeline_levels, key, self.root_path, self.read_only, self.compressed)
        self.__setitem__(key, new_data)

        return self[key]


class HDF5MeasurementData(MeasurementData):

    def __init__(self, h5_file, pipeline_levels, idxes, root_path='experiment', read_only=False, compressed=True):
        super().__init__()
        self.h5_file = h5_file
        self.pll = pipeline_levels
        self.idxes = idxes
        self.root_path = root_path
        self.read_only = read_only
        self.compressed = compressed

        self.load_existing_data()

    def load_existing_data(self):

        # get file handle for write: if member h5_file is already a File object, use as-is
        # otherwise, we assume h5_file is a str/Path and try to open in append mode
        file_is_file_handle = isinstance(self.h5_file, h5py.File)
        fd = self.h5_file if file_is_file_handle else h5py.File(self.h5_file, 'r')

        # query corresponding group in hdf5 file
        path = _hdf5_group_path(self.pll, self.idxes, self.root_path)
        if path in fd:
            h5_dataset = fd[path]
            num_configs = h5py.AttributeManager(h5_dataset)['num_configs']
        else:
            # if group does not (yet) exist, we set num_configs to 0,
            # -> rest of the function (except close) is skipped
            num_configs = 0

        # append data and metadata for all configurations to self
        for config_idx in range(num_configs):
            dataset_configuration = h5_dataset[str(config_idx)]
            att = h5py.AttributeManager(dataset_configuration)
            num_channels = att['num_channels']
            hardware_meta = json.loads(att['global_meta'])
            measurement_meta = json.loads(att['measurement_meta'])
            images = []
            for j in range(num_channels):
                images.append(np.array(dataset_configuration[str(j)]))
            # append to self, do not write back to file (as it already exists)
            self.append(hardware_meta, measurement_meta, images, write_to_file=False)

        # if we have opened a new file object, close it again
        # otherwise (we are using a provided object), leave it
        if not file_is_file_handle:
            fd.close()

    def append(self, hardware_settings=None, measurement_settings=None, data=None, write_to_file=True):

        # add to in-memory copy
        super().append(hardware_settings, measurement_settings, data)

        # we don't actually want to write to file (e.g. when populating the object from already existing data)
        if not write_to_file:
            return

        # error when trying to write to read-only data store
        if self.read_only:
            raise ValueError('Cannot append to read-only HDF5MeasurementData')

        # get file handle for write: if member h5_file is already a File object, use as-is
        # otherwise, we assume h5_file is a str/Path and try to open in append mode
        file_is_file_handle = isinstance(self.h5_file, h5py.File)
        fd = self.h5_file if file_is_file_handle else h5py.File(self.h5_file, 'a')

        # make HDF5 group if it does not exist already (first config in acquisition)
        group_path = _hdf5_group_path(self.pll, self.idxes, self.root_path)
        if group_path not in fd:
            fd.create_group(group_path)
            attrs = h5py.AttributeManager(fd[group_path])
            attrs['num_configs'] = 0

        # get current num of configs -> path for new config
        # then, increment num_configs
        attrs = h5py.AttributeManager(fd[group_path])
        cfg_idx = int(attrs['num_configs'])
        cfg_path = '/'.join([group_path, str(cfg_idx)])
        attrs['num_configs'] = int(attrs['num_configs']) + 1

        # create new group for configuration
        # store num of channels and global + local metadata
        fd.create_group(cfg_path)
        attrs_cfg = h5py.AttributeManager(fd[cfg_path])
        attrs_cfg['num_channels'] = len(data)
        attrs_cfg['measurement_meta'] = json.dumps(measurement_settings, indent=1)
        attrs_cfg['global_meta'] = json.dumps(hardware_settings, indent=1)

        # save channels as actual datasets
        for idx, data_i in enumerate(data):
            path_channel = '/'.join([cfg_path, str(idx)])
            fd.create_dataset(path_channel, data=data_i, compression='gzip' if self.compressed else None)

        # if we have opened a new file object, close it again
        # otherwise (we are using a provided object), leave it
        if not file_is_file_handle:
            fd.close()


class HDF5DataReader(HDF5DataStore):
    """
    reader for experiment data stored as hdf5

    Parameters
    ----------
    fd: h5py File
        the file object to read from, should be opened in read mode
    root_path: str
        name of the root group in fd

    """
    def __init__(self, fd, root_path='experiment'):
        super().__init__(fd, None, root_path, True)


def _hdf5_group_path(pll, idxes, root_name='experiment'):
    path = root_name + '/'
    for i, z in enumerate(zip(pll, idxes)):
        lvl, idx = z
        path = path + '{}{}{}'.format('_' if i != 0 else '', lvl, idx)
    return path


def _path_test():
    pll = ('ov', 'det', 'det2')
    idxes = (1,2)
    print(_hdf5_group_path(pll, idxes))

def main():
    path = 'C:/Users/david/Desktop/msr-test-files/6542d40dcd6ed1833ed868ac060f73a1.h5'
    r = HDF5DataReader(path)
    print(r[(0,22)].measurement_settings)

if __name__ == '__main__':
    _path_test()