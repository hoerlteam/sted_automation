import random
from Util.imspector_util import config_magic
import json

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


# TODO implement me!!!
def set_parameter(params, path, value):
    params.pop(b"is_active")
    params.pop(b"prop_driver")
    params.pop(b"prop_version")
    config = config_magic(path)
    #eval(params str(config)) = value
    params str(config) = value
    return None

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
            set_parameter(params, bytes(k), v)

    def load_from_file(self, path):
        """
        load settings from template file
        :param path:  path to .json file
        :return: None
        """
        # how can i append my settings dict by a json file
        with open(str(path), 'r') as f:
            file = f.read()
        configs_from_json = (json.loads(file))
        for key, value in configs_from_json.items():
            self.settings.update({key: value})
        return None


"""
def main():
    sl = sorted_list()
    sl2 = sorted_list()
    sl.extend([4,2,3])
    sl2.extend([1, 2, 3,0])
    print(sl)
    print(sl2)

    a = [sorted_list() for _ in range(1000)]
    for s in a:
        s.extend([random.random() for _ in range(20)])

    print(a[0])






if __name__ == '__main__':
    main()
"""