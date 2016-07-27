import json
import Util.imspector_util
import numpy as np
import os
import logging


class Sorted_List():
    def __init__(self):
        self.data = list()

    def __str__(self):
        return str(self.data)

    def append(self, a):
        self.data.append(a)
        self.data.sort()

    def extend(self, l):
        self.data.extend(l)
        self.data.sort()

    def __iter__(self):
        return self.data.__iter__()


def set_parameter(params, path, value):
    """
    :param params: parameter object (Imspector)
    :param path: path in the parameters dictionary that shall be set
    :param value: value to set the parameter in the dictionary
    :return: None
    """
    params.pop(b"is_active", None)
    params.pop(b"prop_driver", None)
    params.pop(b"prop_version", None)
    config = Util.imspector_util.config_magic(path)
    # print(config)
    assignment_string = str(value) if not isinstance(value, str) else "'" + value + "'"
    exec("params" + str(config) + " = " + assignment_string)


class Settings:
    def __init__(self):
        self.settings = dict()

    def apply_to_settings_dict(self, params):
        """
        applies settings to the active measurement object
        :param params: ms.setparameters() object
        :return:
        """
        for k, v in self.settings.items():
            set_parameter(params, k, v)

    def load_from_file(self, path):
        """
        load settings from template file
        :param path: path to .json file
        :return: None
        """
        # how can i append my settings dict by a json file
        with open(str(path), 'r') as f:
            file = f.read()
            for key, value in json.loads(file).items():
                self.settings.update({key: value})
        return None

    def set(self, path, value):
        self.settings.update({path: value})

    def __str__(self):
        return str(self.settings)

    def __setitem__(self, key, value):
        self.settings[key] = value

    def clone(self):
        other = Settings()
        other.settings = self.settings.copy()
        return other

    def set_to_coordinates(self, coordination):
        """
        Sets the coordinations of an coordination object on a settings dictionary
        :param coordination: coordination object containing bench_coords, fov_length and offset_coord
        :return: None
        """
        xyz = ("x", "y", "z")
        bench = coordination.get_bench_coords()
        fov = coordination.get_fov_len()
        offset = coordination.get_scan_offset()
        self.set("OlympusIX/scanrange/x/offset", bench[0])
        self.set("OlympusIX/scanrange/y/offset", bench[1])
        self.set("OlympusIX/scanrange/z/off", bench[2])
        for i in range(len(xyz)):
            self.set("ExpControl/scan/range/" + xyz[i] + "/len", fov[i])
        for i in range(len(xyz)):
            self.set("ExpControl/scan/range/" + xyz[i] + "/off", offset[i])



def check_coordinates_valid(coords, min_coords, max_coords):
    return not((np.array(coords) < np.array(min_coords)).any() | (np.array(coords) > np.array(max_coords)).any())


class NameManagement:
    """
    By selecting the path when defining the object two counters and a name is created. Every time when a name is getting
    called by the save function, the counter is increased by 1. It returns: /../path/to/whatever/randomname1.
    Also implemented a STED function to name the STED images with same suffix but sepearte them from the others
    """
    def __init__(self, path, postfix=".msr", separator="_"):
        self.separator = separator
        self.postfix = postfix
        self.path = str(path)
        self.counters = []
        self.prefixes = []
        self.name = str(Util.imspector_util.generate_random_name2())
        self.prefix2idx = dict()

    def add_counter(self, prefix=''):
        self.counters.append(0)
        self.prefixes.append(prefix)
        self.prefix2idx[prefix] = len(self.counters)-1

    def reset_counter(self, idx):
        if (type(idx) == str):
            self.counters[self.prefix2idx[idx]] = 0
        elif (type(idx) == int):
            self.counters[idx] = 0
        else:
            raise Exception("give index or prefix of what you want")

    def get_next_image_name(self, idx):
        if (type(idx) == str):
            return self.get_next_image_name_idx(self.prefix2idx[idx])
        elif (type(idx) == int):
            return self.get_next_image_name_idx(idx)
        else:
            raise Exception("give index or prefix of what you want")

    def get_next_image_name_idx(self, idx):

        self.counters[idx] += 1

        filename = self.name
        for i in range(idx+1):
            filename += self.separator + self.prefixes[i] + str(self.counters[i])

        out = os.path.join(self.path, filename + self.postfix)
        return out

    def get_current_image_name(self):
        return self.name

    # def get_STED_image_name(self):
    #     self.counter2 += 1
    #     out = self.path+"/"+self.name+"STED"+str(self.counter2)+self.postfix
    #     return out


def coordinate_logger(name_object, coordinates):
    """
    Simple wrapper for the logging function adjusted to a certain task.
    Recommended pre-config:
    logging.basicConfig(filename='Insert_filename_for_LOG_here', level=logging.INFO, format='%(message)s')

    :param name_object: NameManagement Object
    :param coordinates: Current coordinates for measurement
    :return: None
    """
    logging.info(("Image: {} at {}".format(name_object.get_current_image_name(), coordinates)))




def main():
   namer = NameManagement("path")
   namer.add_counter("frame")
   namer.add_counter("sted")
   namer.add_counter("onemore")

   print(namer.get_next_image_name("frame"))
   for _ in range(5):
      print(namer.get_next_image_name("sted"))

   namer.reset_counter("sted")
   print(namer.get_next_image_name("frame"))
   for _ in range(5):
      print(namer.get_next_image_name("sted"))


if __name__ == '__main__':
    main()
