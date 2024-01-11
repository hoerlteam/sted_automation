import json
import signal
from math import isclose
import typing as collections
import threading
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
        # signal handling only works on main thread, do nothing if pipeline is running in another
        if threading.current_thread() is threading.main_thread():
            self.old_handler = signal.getsignal(signal.SIGINT)
            signal.signal(signal.SIGINT, self.handler)

    def handler(self, sig, frame):
        self.pipeline.interrupted = True

    def __exit__(self, type, value, traceback):
        if threading.current_thread() is threading.main_thread():
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


def diff_dicts(d1, d2, separator='/'):
    """
    Helper function to compare two complex dictionaries (e.g. Imspector configurations)
    Will return lists of flattened element keys only in one of the dicts
    and keys of differing values.
    """

    # get sets of keys
    keys1 = set(d1.keys())
    keys2 = set(d2.keys())

    # find keys only in one of the two sets
    only1 = keys1.difference(keys2)
    only2 = keys2.difference(keys1)
    diff = set()

    for k in keys1.intersection(keys2):

        # child element with key k is dict in both cases: recurse
        if isinstance(d1[k], dict) and isinstance(d2[k], dict):
            only1_inner, only2_inner, diff_inner = diff_dicts(d1[k], d2[k])

            # add flattened keys with separator to difference sets
            for k_inner in only1_inner:
                only1.add(k + separator + k_inner)
            for k_inner in only2_inner:
                only2.add(k + separator + k_inner)
            for k_inner in diff_inner:
                diff.add(k + separator + k_inner)

        # handle both lists, if at least one element is a dict
        elif (isinstance(d1[k], list) and any((isinstance(v, dict) for v in d1[k])) and
              isinstance(d2[k], list) and any((isinstance(v, dict) for v in d2[k]))):

            # create dummy dicts with index -> value from the lists
            dummy_dict_d1 = {str(i): v for i, v in enumerate(d1[k])}
            dummy_dict_d2 = {str(i): v for i, v in enumerate(d2[k])}
            only1_inner, only2_inner, diff_inner = diff_dicts(dummy_dict_d1, dummy_dict_d2)

            # add flattened keys with separator to difference sets
            for k_inner in only1_inner:
                only1.add(k + separator + k_inner)
            for k_inner in only2_inner:
                only2.add(k + separator + k_inner)
            for k_inner in diff_inner:
                diff.add(k + separator + k_inner)

        # child elements have different type or are "scalar"
        # if both values are float, just check approximate equality
        # NOTE: this will still not catch lists of floats
        elif isinstance(d1[k], float) and isinstance(d2[k], float):
            if not isclose(d1[k], d2[k]):
                diff.add(k)

        # otherwise, just compare
        elif d1[k] != d2[k]:
            diff.add(k)

    # return as sorted lists for easier comparison
    return sorted(only1), sorted(only2), sorted(diff)