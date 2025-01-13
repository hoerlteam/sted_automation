import re
from warnings import warn
import logging
from threading import Semaphore

import numpy as np

try:
    import specpy
except ImportError:
    pass

from autosted.utils.dict_utils import get_path_from_dict
from autosted.utils.parameter_constants import (OFFSET_STAGE_GLOBAL_PARAMETERS, OFFSET_SCAN_PARAMETERS,
                                                 OFFSET_SCAN_GLOBAL_PARAMETERS, FOV_LENGTH_PARAMETERS)


class ParameterSanitizer:

    pattern_parent = re.compile("(?:Internal error: )?Invalid argument for (?:device|parameter) '(.*?)'")
    pattern_last = re.compile("No parameter '(.*?)'")
    # new error message that appeared with the addition of 'Powerswitch'
    unknown_device_error = re.compile("Internal error: Unknown device or parameter '(.*?)'")

    def __init__(self):
        warn('ParameterSanitizer is deprecated', DeprecationWarning)
        self.paths_to_drop = []

    def parse_runtime_error(self, e):        
        lines = e.args[0].split('\n')
        print(lines)

        # check for unknown device error
        m_unknown_device = self.unknown_device_error.match(lines[0])
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
            
    @staticmethod
    def set_parameters_nofail(target, sanitizer, pars):
        done = False
        while not done:
            try:
                sanitizer.sanitize(pars)
                target.set_parameters('', pars)
                done = True
            except RuntimeError as e:
                sanitizer.parse_runtime_error(e)


class ImspectorConnection:

    def __init__(self, imspector=None):

        # make default (local) specpy handle if none is given
        if imspector is None:
            imspector = specpy.get_application()

        self.imspector = imspector
        self.dropped_parameters = set()

        self.logger = logging.getLogger(__name__)

    def set_parameters_recursive(self, where, params, value_type):

        # skip if the parameter caused an error before
        if where in self.dropped_parameters:
            return

        if not isinstance(params, dict):
            # try to set parameter(s), if an error occurs, add to set of parameters to ignore
            try:
                self.imspector.value_at(where, value_type).set(params)
            except RuntimeError:
                if where not in self.dropped_parameters:
                    self.logger.debug('parameter {} cannot be set, ignoring from here on'.format(where))
                    self.dropped_parameters.add(where)
        else:
            # we have a dict of parameters, set all individually
            for k, v in params.items():
                child_key = k if where == '' else where + '/' + k
                self.set_parameters_recursive(child_key, v, value_type)

    def get_current_data(self):
        # get hardware and measurement params (dicts)
        hardware_params = self.imspector.value_at('', specpy.ValueTree.Hardware).get()
        measurement_parameters = self.imspector.active_measurement().parameters('')
        # get list of all stacks in currently active measurement / configuration
        stack_data = []
        for name in self.imspector.active_measurement().stack_names():
            stack_data.append(np.copy(self.imspector.active_measurement().stack(name).data()))
        return hardware_params, measurement_parameters, stack_data

    @staticmethod
    def get_n_channels(parameters):
        return len(get_path_from_dict(parameters, 'ExpControl/measurement/channels', False))
    
    def make_measurement_from_task(self, task):

        ms = self.imspector.create_measurement()
        self.set_parameters_in_measurement(ms, task)

    def set_parameters_in_measurement(self, ms, task):

        measurement_updates, hardware_updates = task
        # we do the update twice to also set grayed-out values
        self.set_parameters_recursive('', measurement_updates, specpy.ValueTree.Measurement)
        if len(hardware_updates) > 0:
            self.set_parameters_recursive('', hardware_updates, specpy.ValueTree.Hardware)
        self.set_parameters_recursive('', measurement_updates, specpy.ValueTree.Measurement)
        if len(hardware_updates) > 0:
            self.set_parameters_recursive('', hardware_updates, specpy.ValueTree.Hardware)

        # NOTE: sync axis seems to jump back to frame after setting
        # if we want lines, we manually re-set just that one parameter
        # FIXME: this causes weird problems in xz cut followed by any other image
        if get_path_from_dict(measurement_updates, 'Measurement/axes/num_synced', False) == 1:
            ms.set_parameters('Measurement/axes/num_synced', 1)

    def make_configuration_from_task(self, task):

        # clone configuration in current measurement
        ms = self.imspector.active_measurement()
        ac = ms.active_configuration()
        ac = ms.clone(ac)
        ms.activate(ac)

        self.set_parameters_in_measurement(ms, task)

    def run_current_measurement(self):

        # get last configuration of active measurement
        ms = self.imspector.active_measurement()
        ms.activate(ms.configuration(ms.number_of_configurations()-1))
        
        # get acquisition coordinates / fov and log for debug
        params = ms.parameters('')
        offsets_stage = [get_path_from_dict(params, path, False) for path in OFFSET_STAGE_GLOBAL_PARAMETERS]
        offsets_scan = [get_path_from_dict(params, path, False) for path in OFFSET_SCAN_PARAMETERS]
        offsets_scan_global = [get_path_from_dict(params, path, False) for path in OFFSET_SCAN_GLOBAL_PARAMETERS]
        fov_length = [get_path_from_dict(params, path, False) for path in FOV_LENGTH_PARAMETERS]
        self.logger.debug('running acquisition:')
        self.logger.debug('stage offsets: {}'.format(offsets_stage))
        self.logger.debug('scan offsets: {}'.format(offsets_scan))
        self.logger.debug('scan offsets global: {}'.format(offsets_scan_global))
        self.logger.debug('FOV length: {}'.format(fov_length))

        # actually run
        self.imspector.run(ms)

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

    def save_current_measurement(self, path):
        ms = self.imspector.active_measurement()
        ms.save_as(path)

    def close_current_measurement(self):
        ms = self.imspector.active_measurement()
        self.imspector.close(ms)

def get_active_measurement_safe(imspector):
    # Getting active measurement from Imspector fails with RuntimeError
    # if no measurement is open or if none is selected (e.g. after closing)
    # This function catches the exception and returns None instead
    try:
        return imspector.active_measurement()
    except RuntimeError:
        return None

def get_current_stage_coords(im=None):

    if im is None:
        im = specpy.get_application()

    ms = get_active_measurement_safe(im)
    need_temp_measurement = ms is None
    if need_temp_measurement:
        im.create_measurement()
        ms = im.active_measurement()

    coords = [ms.parameters(path) for path in OFFSET_STAGE_GLOBAL_PARAMETERS]

    if need_temp_measurement:
        im.close(ms)

    return coords
