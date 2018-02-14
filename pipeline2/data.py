import pipeline2
import h5py
import json
from collections import defaultdict

class RichData:
    """
    wrapper for data and settings of a measurement
    holds hardware ('global') settings and per measurement settings
    for each parameter set and a list of associated data
    """

    def __init__(self):
        self.globalSettings = []
        self.measurementSettings = []
        self.data = []

    # TODO: remove defaults?
    def append(self, globalSettings=None, measurementSettings=None, data=None):
        self.globalSettings.append(globalSettings)
        self.measurementSettings.append(measurementSettings)
        self.data.append(data)

    @property
    def numConfigurations(self):
        return len(self.data)

    def numImages(self, n):
        if n < self.numConfigurations:
            return len(self.data[n])
        else:
            return 0


class HDF5DataStore(defaultdict):

    def __init__(self, fd, pipeline_levels):
        super().__init__(lambda idx : HDF5RichData(fd, pipeline_levels, idx ))
        self.fd = fd
        self.pll = pipeline_levels

        # TODO: write some general info (pipeline levels, etc. ) to root ('experiment') group

    def __missing__(self, key):
        ret = self[key] = self.default_factory(key)
        return ret


class HDF5RichData(RichData):

    def __init__(self, fd, pipeline_levels, idxes):
        super(HDF5RichData, self).__init__()
        self.fd = fd
        self.pll = pipeline_levels
        self.idxes = idxes

    def append(self, globalSettings=None, measurementSettings=None, data=None):
        super(HDF5RichData, self).append(globalSettings, measurementSettings, data)

        # make HDF5 group if it does not exist already (first config in acquisition)
        group_path = _hdf5_group_path(self.pll, self.idxes)
        if not group_path in self.fd:
            self.fd.create_group(group_path)
            attrs = h5py.AttributeManager(self.fd[group_path])
            attrs['num_configs'] = 0

        # get current num of configs -> path for new config
        # then, increment num_configs
        attrs = h5py.AttributeManager(self.fd[group_path])
        cfg_idx = int(attrs['num_configs'])
        cfg_path = '/'.join([group_path, str(cfg_idx)])
        attrs['num_configs'] = int(attrs['num_configs']) + 1

        # create new group for configuration
        # store num of channels and global + local metadata
        self.fd.create_group(cfg_path)
        attrs_cfg = h5py.AttributeManager(self.fd[cfg_path])
        attrs_cfg['num_channels'] = len(data)
        attrs_cfg['measurement_meta'] = json.dumps(measurementSettings, indent=1)
        attrs_cfg['global_meta'] = json.dumps(globalSettings, indent=1)

        # save channels as actual datasets
        for idx, data_i in enumerate(data):
            path_channel = '/'.join([cfg_path, str(idx)])
            self.fd.create_dataset(path_channel, data_i)


def _hdf5_group_path(pll, idxes, root_name='experiment'):
    path = root_name + '/'
    for i, z in enumerate(zip(pll.levels, idxes)):
        lvl, idx = z
        path = path + '{}{}{}'.format('_' if i!=0 else '', lvl, idx)
    return path


def main():
    pll = pipeline2.PipelineLevels('ov', 'det', 'det2')
    idxes = (1,2)
    print(_hdf5_group_path(pll, idxes))

if __name__ == '__main__':
    main()