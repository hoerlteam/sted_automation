from ..util import update_dicts, filter_dict
import numpy as np
import time
from unittest.mock import MagicMock
from ..data import RichData

# TODO: remove in production?
try:
    from specpy import Imspector
except ImportError:
    Imspector = MagicMock()



class MockImspectorConnection():
    def __init__(self):
        self.getCurrentData = MagicMock(return_value=RichData())
        self.makeMeasurementFromTask = MagicMock(return_value=None)
        self.makeConfigurationFromTask = MagicMock(return_value=None)
        self.runCurrentMeasurement = MagicMock(return_value=None)
        self.saveCurrentMeasurement = MagicMock(return_value=None)
        self.closeCurrentMeasurement = MagicMock(return_value=None)


class ImspectorConnection():
    def __init__(self, im):
        self.im = im

    def getCurrentData(self):
        globalParams = self.im.parameters('')
        measParameters = self.im.active_measurement().parameters('')
        data = []
        for name in self.im.active_measurement().stack_names():
            data.append(np.copy(self.im.active_measurement().stack(name).data()))
        return globalParams, measParameters, data

    def makeMeasurementFromTask(self, task, halfDelay=0.0):
        ms = self.im.create_measurement()
        measUpdates, confUpdates = task
        measUpdates = update_dicts(*measUpdates)
        confUpdates = update_dicts(*confUpdates)

        # we do the update twice to also set grayed-out values
        ms.set_parameters('', measUpdates)
        self.im.set_parameters('', confUpdates)
        # wait if requested
        time.sleep(halfDelay)
        ms.set_parameters('', measUpdates)
        self.im.set_parameters('', confUpdates)
        # wait again if requested
        time.sleep(halfDelay)

        # NB: sync axis seems to jump back to frame after setting
        # if we want lines, we manually re-set just that one parameter

        if filter_dict(measUpdates, 'Measurement/axes/num_synced', False) == 1:
            ms.set_parameters('Measurement/axes/num_synced', 1)

    def makeConfigurationFromTask(self, task, halfDelay = 0.0):
        ms = self.im.active_measurement()
        ac = ms.active_configuration()
        ac = ms.clone(ac)
        ms.activate(ac)

        measUpdates, confUpdates = task
        measUpdates = update_dicts(*measUpdates)
        confUpdates = update_dicts(*confUpdates)

        # we do the update twice to also set grayed-out values
        ms.set_parameters('', measUpdates)
        self.im.set_parameters('', confUpdates)
        # wait if requested
        time.sleep(halfDelay)
        ms.set_parameters('', measUpdates)
        self.im.set_parameters('', confUpdates)
        # wait again if requested
        time.sleep(halfDelay)

        # NB: sync axis seems to jump back to frame after setting
        # if we want lines, we manually re-set just that one parameter

        if filter_dict(measUpdates, 'Measurement/axes/num_synced', False) == 1:
            ms.active_configuration().set_parameters('Measurement/axes/num_synced', 1)

    def runCurrentMeasurement(self, task=None):
        
        ms = self.im.active_measurement()
        ms.activate(ms.configuration(ms.number_of_configurations()-1))
        
        # re-check num_synched
        if task is not None:
            measUpdates, confUpdates = task
            measUpdates = update_dicts(*measUpdates)
            confUpdates = update_dicts(*confUpdates)
            if filter_dict(measUpdates, 'Measurement/axes/num_synced', False) == 1:
                ms.set_parameters('Measurement/axes/num_synced', 1)
                ms.configuration(ms.number_of_configurations()-1).set_parameters('Measurement/axes/num_synced', 1)
                
        self.im.run(ms)

    def saveCurrentMeasurement(self, path):
        ms = self.im.active_measurement()
        ms.save_as(path)

    def closeCurrentMeasurement(self):
        ms = self.im.active_measurement()
        self.im.close(ms)


def get_current_stage_coords(im=None):
    if im is None:
        im = Imspector()

    im.create_measurement()
    ms = im.active_measurement()

    coords = [ms.parameters('ExpControl/scan/range/offsets/coarse/' + c + '/g_off') for c in 'xyz']

    im.close(ms)

    return coords