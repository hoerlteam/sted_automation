import json
import Util.imspector_util

class sorted_list():
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


def params(config):
    pass


def set_parameter(params, path, value):
    """
    :param params: parameter object (Imspector)
    :param path: path in the parameters dictionary that shall be set
    :param value: value to set the parameter in the dictionary
    :return: Not implemented
    """
    params.pop(b"is_active", None)
    params.pop(b"prop_driver", None)
    params.pop(b"prop_version", None)
    config = Util.imspector_util.config_magic(path)
    #print(config)
    assignment_string = str(value) if not isinstance(value, str) else "'" + value + "'"
    exec("params" + str(config) + " = " + assignment_string )


class Settings():
    def __init__(self):
        self.settings = dict()

    def apply_to_settings_dict(self, params):
        """
        applies settings to the active measurement object
        :param params: ms.setparameters() object
        :return:
        """
        for k,v in self.settings.items():
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
        return(other)


class Coordination:
    """
    This class writes th parameters for the global bench coordinates, the lenght of the field of view
    and the scan offset in a list. [bench_coordinates, fov_lenght, scan_offset].
    """
    def __init__(self, ms):
        bench_coords = (self.bench_coords_snapshot(ms))
        fov_len = (self.fov_len_snapshot(ms))
        offset_coords = (self.scan_offset_coords_snapshot(ms))
        self.coordinates = (bench_coords, fov_len, offset_coords)

    def bench_coords_snapshot(self, ms):
        x = ms.parameter("OlympusIX/scanrange/x/offset")
        y = ms.parameter("OlympusIX/scanrange/y/offset")
        z = ms.parameter("OlympusIX/scanrange/z/off")
        return [x, y, z]

    def fov_len_snapshot(self, ms):
        x = ms.parameter("ExpControl/scan/range/x/len")
        y = ms.parameter("ExpControl/scan/range/y/len")
        z = ms.parameter("ExpControl/scan/range/z/len")
        return [x, y, z]

    def scan_offset_coords_snapshot(self, ms):
        x = ms.parameter("ExpControl/scan/range/x/off")
        y = ms.parameter("ExpControl/scan/range/y/off")
        z = ms.parameter("ExpControl/scan/range/z/off")
        return [x, y, z]

    def get_bench_coords(self):
        """
        :return: returns the coordinates of the bench in form [x, y, z]
        """
        return self.coordinates[1]

    def get_fov_len(self):
        """
        :return: returns the length of the the fov in form [x, y, z]
        """
        return self.coordinates[2]

    def get_scan_offset(self):
        """
        :return: returns the scan-offset in form of [x, y, z]
        """
        return self.coordinates[3]




def main():
    a = Settings()
    a.load_from_file("./test.json")
    b = a.clone()

    print(a)
    print(b)





if __name__ == '__main__':
    main()
