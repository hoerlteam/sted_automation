import json
import re
from warnings import warn
import logging
from threading import Semaphore

from ..util import update_dicts, filter_dict
from ..ressources import dummy_measurements
import numpy as np
import time
from unittest.mock import MagicMock
from ..data import RichData

# TODO: remove in production?
try:
    from specpy import Imspector
    import specpy
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
        self.dropped_parameters = set()

    def set_parameters_recursive(self, where, params, value_type):
        if not isinstance(params, dict):
            try:
                self.im.value_at(where, value_type).set(params)
            except RuntimeError as e:
                if where not in self.dropped_parameters:
                    logging.debug('WARNING: parameter {} cannot be set, ignoring from here on'.format(where))
                    self.dropped_parameters.add(where)
        else:
            for k,v in params.items():
                self.set_parameters_recursive(k if where == '' else where + '/' + k, v, value_type)

    def withVerbose(self, verbose=True):
        self.verbose = verbose
        return self
        
    def getCurrentData(self):
        globalParams = self.im.value_at('').get()
        measParameters = self.im.active_measurement().parameters('')
        data = []
        for name in self.im.active_measurement().stack_names():
            data.append(np.copy(self.im.active_measurement().stack(name).data()))
        return globalParams, measParameters, data

    @staticmethod
    def get_n_channels(parameters):
        return len(filter_dict(parameters, 'ExpControl/measurement/channels', False))
    
    def makeMeasurementFromTask(self, task, halfDelay=0.0):
        # FIXME: check if all the delays are really necessary
        
        # TODO: hacky fix for various channel numbers
        ms = self.im.create_measurement()
        #time.sleep(halfDelay)
        measUpdates, confUpdates = task
        measUpdates = update_dicts(*measUpdates)
        confUpdates = update_dicts(*confUpdates)
        
        #n_channels = ImspectorConnection.get_n_channels(measUpdates)
        #print('DEBUG: n_channels', n_channels)
        
        #ms = self.im.open(dummy_measurements[n_channels])
        #self.im.activate(ms)
        
        # remove old stacks?
        #ac = ms.active_configuration()
        #ac2 = ms.clone(ac)
        #ms.activate(ac2)
        #ms.remove(ac)

        # we do the update twice to also set grayed-out values
        #set_parameters_nofail(ms, self.sanitizer_ms, measUpdates)
        self.set_parameters_recursive('', measUpdates, specpy.ValueTree.Measurement)
        time.sleep(halfDelay)
        #set_parameters_nofail(self.im, self.sanitizer_im, confUpdates)
        self.set_parameters_recursive('', confUpdates, specpy.ValueTree.Hardware)
        # wait if requested
        time.sleep(halfDelay)
        #set_parameters_nofail(ms, self.sanitizer_ms, measUpdates)
        self.set_parameters_recursive('', measUpdates, specpy.ValueTree.Measurement)
        time.sleep(halfDelay)
        #set_parameters_nofail(self.im, self.sanitizer_im, confUpdates)
        self.set_parameters_recursive('', confUpdates, specpy.ValueTree.Hardware)
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
        #n_channels_existing = ImspectorConnection(ac.parameters(''))
                
        measUpdates, confUpdates = task
        measUpdates = update_dicts(*measUpdates)
        confUpdates = update_dicts(*confUpdates)
        
        _channels = ImspectorConnection.get_n_channels(measUpdates)
        
        #if (n_channels_existing == n_channels):
        ac = ms.clone(ac)
        ms.activate(ac)
        # else:
        #     warn('Number of channels in configurations differs. Only the last configurations will be saved as MSR')
        #     ms = self.im.open(dummy_measurements[n_channels])
        #     self.im.activate(ms)
        #     # remove old stacks?
        #     ac = ms.active_configuration()
        #     ac2 = ms.clone(ac)
        #     ms.activate(ac2)
        #     ms.remove(ac)

        # we do the update twice to also set grayed-out values
        #set_parameters_nofail(ms, self.sanitizer_ms, measUpdates)
        self.set_parameters_recursive('', measUpdates, specpy.ValueTree.Measurement)
        time.sleep(halfDelay)
        #set_parameters_nofail(self.im, self.sanitizer_im, confUpdates)
        self.set_parameters_recursive('', confUpdates, specpy.ValueTree.Hardware)
        # wait if requested
        time.sleep(halfDelay)
        #set_parameters_nofail(ms, self.sanitizer_ms, measUpdates)
        self.set_parameters_recursive('', measUpdates, specpy.ValueTree.Measurement)
        time.sleep(halfDelay)
        #set_parameters_nofail(self.im, self.sanitizer_im, confUpdates)
        self.set_parameters_recursive('', confUpdates, specpy.ValueTree.Hardware)
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
                par, 'ExpControl/scan/range/coarse_{}/g_off'.format(c), False) for c in ['x', 'y', 'z']], dtype=float)
            offsScan = np.array([filter_dict(
                par, 'ExpControl/scan/range/{}/off'.format(c), False) for c in ['x', 'y', 'z']], dtype=float)
            offsGlobal = np.array([filter_dict(
                par, 'ExpControl/scan/range/{}/g_off'.format(c), False) for c in ['x', 'y', 'z']], dtype=float)
            
            print('running acquisition:')
            print('stage offsets: {}'.format(offsStage))
            print('scan offsets: {}'.format(offsScan))
            print('scan offsets global: {}'.format(offsGlobal))
            
        self.im.run(ms)

    @staticmethod
    def blocking_imspector_run(imspector, measurement):
        '''
        since imspector.run(measurement) was buggy in 16.3.10599-1948 and did not block,
        this wrapps it into a blocking call.

        also catches the 'bad cast' exception that seems to always happen
        '''

        warn('blocking_imspector_run is deprecated and no longer necessary', DeprecationWarning)

        sem = Semaphore()
        sem.acquire()
        imspector.connect_end(sem.release, 0)

        try:
            imspector.run(measurement)
        except RuntimeError as e:
            # bad cast error is always thrown, but measurement seems to work nonetheless
            # in that specific case, we ignore the error
            if str(e) == 'bad cast':
                pass
            else:
                raise

        sem.acquire()
        imspector.disconnect_end(sem.release, 0)


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

    coords = [ms.parameters('ExpControl/scan/range/coarse_' + c + '/g_off') for c in 'xyz']

    im.close(ms)

    return coords