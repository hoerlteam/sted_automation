import json
import signal
import collections
from copy import deepcopy


class DelayedKeyboardInterrupt():
    """
    context manager to allow finishing of one acquisition loop
    before quitting queue due to KeyboardInterrupt

    modified from https://stackoverflow.com/a/21919644
    """

    def __init__(self, pipeline):
        self.pipeline = pipeline

    def __enter__(self):
        self.old_handler = signal.getsignal(signal.SIGINT)
        signal.signal(signal.SIGINT, self.handler)

    def handler(self, sig, frame):
        self.pipeline.interrupted = True

    def __exit__(self, type, value, traceback):
        signal.signal(signal.SIGINT, self.old_handler)

def dump_JSON(d, path):
    """
    helper function to dump a dict to a file given by path as JSON
    """
    with open(path, 'w') as fd:
        json.dump(d, fd, indent=2)


def update_dicts(*dicts):
    if len(dicts) == 0:
        return {}

    first = dicts[0]
    if len(dicts) < 2:
        return deepcopy(first)
    else:
        second = dicts[1]
        return update_dicts(update_dict_pair(first, second), *dicts[2:])


def update_dict_pair(d1, d2):
    res = deepcopy(d1)
    for k, v in d2.items():
        if (isinstance(v, collections.Mapping)):
            res[k] = update_dict_pair(res.get(k) if isinstance(res.get(k, None), collections.Mapping) else {}, v)
        else:
            res[k] = v
    return res


def remove_filter_from_dict(d, flt, sep='/'):
    flt_strp = flt.strip(sep)
    flts = flt_strp.split(sep)
    fst_flt = flts[0]

    if fst_flt == '':
        return None

    if len(flts) == 1:
        if (isinstance(d, collections.Sequence) and fst_flt.isdigit()):
            try:
                cpy = deepcopy(d)
                cpy.pop(int(fst_flt))
                return cpy if len(cpy) > 0 else None
            except IndexError:
                return deepcopy(d)
        elif (isinstance(d, collections.Mapping)):
            try:
                cpy = deepcopy(d)
                del cpy[fst_flt]
                return cpy if len(cpy) > 0 else None
            except KeyError:
                return deepcopy(d)
        else:
            return deepcopy(d)

    if (isinstance(d, collections.Sequence) and fst_flt.isdigit()):
        try:
            res = remove_filter_from_dict(d[int(fst_flt)], sep.join(flts[1:]))
            cpy = deepcopy(d)
            if res is None:
                del cpy[int(fst_flt)]
            else:
                cpy[int(fst_flt)] = res
            return cpy if len(cpy) > 0 else None
        except IndexError:
            return deepcopy(d)

    elif (isinstance(d, collections.Mapping)):
        try:
            res = remove_filter_from_dict(d[fst_flt], sep.join(flts[1:]))
            cpy = deepcopy(d)
            if res is None:
                del cpy[fst_flt]
            else:
                cpy[fst_flt] = res
            return cpy if len(cpy) > 0 else None
        except KeyError:
            return deepcopy(d)
    else:
        return None


def filter_dict(d, flt, keepStructure=True, sep='/'):
    flt_strp = flt.strip(sep)
    flts = flt_strp.split(sep)

    fst_flt = flts[0]

    if fst_flt == '':
        return d

    if len(flts) == 1:
        if (isinstance(d, collections.Sequence) and fst_flt.isdigit()):
            try:
                return [d[int(fst_flt)]] if keepStructure else d[int(fst_flt)]
            except IndexError:
                return None
        elif (isinstance(d, collections.Mapping)):
            try:
                return {fst_flt: d[fst_flt]} if keepStructure else d[fst_flt]
            except KeyError:
                return None
        else:
            return None

    if (isinstance(d, collections.Sequence) and fst_flt.isdigit()):
        try:
            res = filter_dict(d[int(fst_flt)], sep.join(flts[1:]), keepStructure)
            if res is None:
                return None
            return [res] if keepStructure else res
        except IndexError:
            return None
    elif (isinstance(d, collections.Mapping)):
        try:
            res = filter_dict(d[fst_flt], sep.join(flts[1:]), keepStructure)
            if res is None:
                return None
            return {fst_flt: res} if keepStructure else res
        except KeyError:
            return None
    else:
        return None


def gen_json(data, path, sep='/'):
    path_strp = path.strip(sep)
    paths = path_strp.split(sep)
    fst_path = paths[0]

    if fst_path == '':
        return data
    else:
        return {fst_path: gen_json(data, sep.join(paths[1:]))}