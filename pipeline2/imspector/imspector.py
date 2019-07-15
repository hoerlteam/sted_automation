import json
import re
from warnings import warn

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

# new error message that appeared with the addition of 'Powerswitch'
_unknown_device_error = re.compile("Internal error: Unknown device or parameter '(.*?)'")

class ParameterSanitizer:
    
    def __init__(self):
        self.paths_to_drop = []
        self.pattern_parent = re.compile("(?:Internal error: )?Invalid argument for (?:device|parameter) '(.*?)'")
        self.pattern_last = re.compile("No parameter '(.*?)'")
        
    def parse_runtime_error(self, e):        
        lines = e.args[0].split('\n')
        print(lines)

        # check for unknown device error
        m_unknown_device = _unknown_device_error.match(lines[0])
        if m_unknown_device:
            # add device to ignored list
            par_dev_unknown = [m_unknown_device.groups()[0]]
            self.paths_to_drop.append(par_dev_unknown)

            # notify user and quit further parsing of this error
            print('WARNING: parameter {} cannot be set, ignoring from here on'.format(par_dev_unknown))
            return

        pars = [self.pattern_parent.match(l).groups()[0] for l in lines[:-1]]
        m = self.pattern_last.match(lines[-1])
        if m:
            pars.append(m.groups()[0])
            
        self.paths_to_drop.append(pars)
        print('WARNING: parameter {} cannot be set, ignoring from here on'.format(pars))
        
    def sanitize(self, p):
        for par in self.paths_to_drop:
            ParameterSanitizer.del_recursive(p, par)

    @staticmethod
    def del_recursive(d, params):

        # ignore empty, but warn
        if len(params)==0:
            warn('WARNING: tried to remove empty parameter (this is probably okay, but should not happen - look into it!)')
            return

        for i in range(len(params) - 1):
            if params[i] in d:
                d = d[params[i]]
            else:
                break
        if params[-1] in d:
            del d[params[-1]]
            
            
def set_parameters_nofail(target, sanitizer, pars):
    done = False
    while not done:
        try:
            sanitizer.sanitize(pars)
            target.set_parameters('', pars)
            done = True
        except RuntimeError as e:
            sanitizer.parse_runtime_error(e) 


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
        self.verbose = False
        self.sanitizer_im = ParameterSanitizer()
        self.sanitizer_ms = ParameterSanitizer()

    def withVerbose(self, verbose=True):
        self.verbose = verbose
        return self
        
    def getCurrentData(self):
        globalParams = self.im.parameters('')
        measParameters = self.im.active_measurement().parameters('')
        data = []
        for name in self.im.active_measurement().stack_names():
            data.append(np.copy(self.im.active_measurement().stack(name).data()))
        return globalParams, measParameters, data

    def makeMeasurementFromTask(self, task, halfDelay=0.0):
        # FIXME: check if all the delays are really necessary
        ms = self.im.create_measurement()
        time.sleep(halfDelay)
        measUpdates, confUpdates = task
        measUpdates = update_dicts(*measUpdates)
        confUpdates = update_dicts(*confUpdates)

        # we do the update twice to also set grayed-out values
        set_parameters_nofail(ms, self.sanitizer_ms, measUpdates)
        time.sleep(halfDelay)
        set_parameters_nofail(self.im, self.sanitizer_im, confUpdates)
        # wait if requested
        time.sleep(halfDelay)
        set_parameters_nofail(ms, self.sanitizer_ms, measUpdates)
        time.sleep(halfDelay)
        set_parameters_nofail(self.im, self.sanitizer_im, confUpdates)
        # wait again if requested
        time.sleep(halfDelay)

        # NB: sync axis seems to jump back to frame after setting
        # if we want lines, we manually re-set just that one parameter
        # FIXME: this causes weird problems in xz cut followed by any other image

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
        set_parameters_nofail(ms, self.sanitizer_ms, measUpdates)
        time.sleep(halfDelay)
        set_parameters_nofail(self.im, self.sanitizer_im, confUpdates)
        # wait if requested
        time.sleep(halfDelay)
        set_parameters_nofail(ms, self.sanitizer_ms, measUpdates)
        time.sleep(halfDelay)
        set_parameters_nofail(self.im, self.sanitizer_im, confUpdates)
        # wait again if requested
        time.sleep(halfDelay)

        # NB: sync axis seems to jump back to frame after setting
        # if we want lines, we manually re-set just that one parameter
        # FIXME: this causes weird problems in xz cut followed by any other image

        if filter_dict(measUpdates, 'Measurement/axes/num_synced', False) == 1:
            ms.set_parameters('Measurement/axes/num_synced', 1)

    def runCurrentMeasurement(self, task=None):
        
        ms = self.im.active_measurement()
        ms.activate(ms.configuration(ms.number_of_configurations()-1))
        
        # re-check num_synched
        # TODO: re-activate ?
        # FIXME: this causes weird problems in xz cut followed by any other image
        '''
        if task is not None:
            measUpdates, confUpdates = task
            measUpdates = update_dicts(*measUpdates)
            confUpdates = update_dicts(*confUpdates)
            if filter_dict(measUpdates, 'Measurement/axes/num_synced', False) == 1:
                ms.set_parameters('Measurement/axes/num_synced', 1)
                ms.configuration(ms.number_of_configurations()-1).set_parameters('Measurement/axes/num_synced', 1)
        '''
        
        if self.verbose:
            par = ms.parameters('')
            offsStage = np.array([filter_dict(
                par, 'ExpControl/scan/range/offsets/coarse/{}/g_off'.format(c), False) for c in ['x', 'y', 'z']], dtype=float)
            offsScan = np.array([filter_dict(
                par, 'ExpControl/scan/range/{}/off'.format(c), False) for c in ['x', 'y', 'z']], dtype=float)
            offsGlobal = np.array([filter_dict(
                par, 'ExpControl/scan/range/{}/g_off'.format(c), False) for c in ['x', 'y', 'z']], dtype=float)
            
            print('running acquisition:')
            print('stage offsets: {}'.format(offsStage))
            print('scan offsets: {}'.format(offsScan))
            print('scan offsets global: {}'.format(offsGlobal))
            
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