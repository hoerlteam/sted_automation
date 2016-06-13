import json
import imspector_util as iu

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


# TODO document me!!!
def set_parameter(params, path, value):
    params.pop(b"is_active")
    params.pop(b"prop_driver")
    params.pop(b"prop_version")
    config = iu.config_magic(path)
    eval(params + str(config) + " = " + str(value))


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
        :param path:  path to .json file
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




def main():
    a = Settings()
    a.load_from_file("./test.json")
    b = a.clone()

    print(a)
    print(b)





if __name__ == '__main__':
    main()
