import pipeline2
import h5py
import json
import re
from collections import defaultdict
import numpy as np

class H5DataReader:
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
        self.fd = fd
        self.root_path = root_path
        self.levels = h5py.AttributeManager(self.fd[root_path])['levels'].split(',')
        self.idxes = self._get_indices()

    def _get_indices(self):
        """
        get all the numerical indices of data in the hdf5 file
        :return: list if index tuples
        """

        p = re.compile('(\\d*)(?:_){,1}'.join(map(lambda l : '(?:{}){{,1}}'.format(l), self.levels)) + '(\\d*)')
        idxes = []

        for k in self.fd[self.root_path].keys():
            if not p.match(k):
                continue
            g = p.match(k).groups()
            g = tuple([int(gi) for gi in g if gi != ''])
            idxes.append(g)
        return idxes

    def get_data(self, idx):
        """
        get the RichData with the given numerical index or None, if it does not exist
        :param idx: int-tuple, the index of the data to access
        :return:
        """
        if idx not in self.idxes:
            return None
        pll = pipeline2.PipelineLevels(*self.levels)
        path = _hdf5_group_path(pll, idx, self.root_path)
        ds = self.fd[path]

        num_configs = h5py.AttributeManager(ds)['num_configs']

        res = RichData()
        for i in range(num_configs):
            ds_i = ds[str(i)]
            att = h5py.AttributeManager(ds_i)
            num_channels = att['num_channels']
            global_meta = json.loads(att['global_meta'])
            measurement_meta = json.loads(att['measurement_meta'])
            imgs = []
            for j in range(num_channels):
                imgs.append(np.array(ds_i[str(j)]))
            res.append(global_meta, measurement_meta, imgs)
        return res


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

    def __init__(self, fd, pipeline_levels, root_path='experiment'):
        super().__init__(lambda idx : HDF5RichData(fd, pipeline_levels, idx , root_path))
        self.fd = fd
        self.pll = pipeline_levels

        if not root_path in self.fd:
            self.fd.create_group(root_path)
        attrs = h5py.AttributeManager(self.fd[root_path])
        attrs['levels'] = ','.join([str(lvl) for lvl in self.pll.levels])

    def __missing__(self, key):
        ret = self[key] = self.default_factory(key)
        return ret


class HDF5RichData(RichData):

    def __init__(self, fd, pipeline_levels, idxes, root_path='experiment'):
        super().__init__()
        self.fd = fd
        self.pll = pipeline_levels
        self.idxes = idxes
        self.root_path = root_path

    def append(self, globalSettings=None, measurementSettings=None, data=None):
        super().append(globalSettings, measurementSettings, data)

        # make HDF5 group if it does not exist already (first config in acquisition)
        group_path = _hdf5_group_path(self.pll, self.idxes, self.root_path)
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
            self.fd.create_dataset(path_channel, data=data_i)


def _hdf5_group_path(pll, idxes, root_name='experiment'):
    path = root_name + '/'
    for i, z in enumerate(zip(pll.levels, idxes)):
        lvl, idx = z
        path = path + '{}{}{}'.format('_' if i!=0 else '', lvl, idx)
    return path


def _path_test():
    pll = pipeline2.PipelineLevels('ov', 'det', 'det2')
    idxes = (1,2)
    print(_hdf5_group_path(pll, idxes))

def main():
    path = '/Volumes/cooperation_data/TobiasRagoczy_StamLab/DavidHoerl/20180226_pipeline2test/70e4cfc46f335a6f75066cfbbf65f8d9.h5'
    with h5py.File(path, 'r') as fd:
        r = H5DataReader(fd)
        print(r.get_data((0,0)).measurementSettings)

if __name__ == '__main__':
    main()