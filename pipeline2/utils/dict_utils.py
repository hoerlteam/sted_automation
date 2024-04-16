import json
import typing as collections
from copy import deepcopy
from math import isclose


def dump_json_to_file(d, path):
    """
    helper function to save a dict in JSON-format to a file given by path
    """
    with open(path, 'w') as fd:
        json.dump(d, fd, indent=2)


def update_dicts(*dicts):
    """
    update / merge multiple dicts
    this will work in a reduce-like fashion, merging the first two, then the result with the third, ...
    later updates will overwrite values in earlier dicts
    """
    if len(dicts) == 0:
        return {}

    first = dicts[0]
    if len(dicts) < 2:
        return deepcopy(first)
    else:
        second = dicts[1]
        return update_dicts(update_dict_pair(first, second), *dicts[2:])


def update_dict_pair(dict_old, dict_new):
    """
    update / merge two dicts
    the resulting dict will be the union of the existing dicts, but in case of overlapping keys
    the second (new) dict's values will overwrite the values from the first (old) dict
    """
    res = deepcopy(dict_old)
    for k, v in dict_new.items():
        if isinstance(v, collections.Mapping):
            res[k] = update_dict_pair(res.get(k) if isinstance(res.get(k, None), collections.Mapping) else {}, v)
        else:
            res[k] = v
    return res


def remove_path_from_dict(d, path, sep='/'):
    """
    delete elements from a nested dict based on a XPath-like path
    can handle both nested dicts/mappings and nested lists/sequences
    """

    filter_paths = path.strip(sep).split(sep)
    first_filter = filter_paths[0]

    # we have reached the end of the path, return None -> will be discarded in outer recursive calls
    if first_filter == '':
        return None

    # end of recursion at sequence / dict: make copy and remove key if lowest level is a dict or index if lowest level is a sequence
    if len(filter_paths) == 1:
        if isinstance(d, collections.Sequence) and first_filter.isdigit():
            try:
                cpy = deepcopy(d)
                cpy.pop(int(first_filter))
                return cpy if len(cpy) > 0 else None
            except IndexError:
                return deepcopy(d)
        elif isinstance(d, collections.Mapping):
            try:
                cpy = deepcopy(d)
                del cpy[first_filter]
                return cpy if len(cpy) > 0 else None
            except KeyError:
                return deepcopy(d)
        else:
            return deepcopy(d)

    # more than one layer remaining, currently at a sequence
    # replace with result of a recursive call
    if isinstance(d, collections.Sequence) and first_filter.isdigit():
        try:
            res = remove_path_from_dict(d[int(first_filter)], sep.join(filter_paths[1:]))
            cpy = deepcopy(d)
            if res is None:
                del cpy[int(first_filter)]
            else:
                cpy[int(first_filter)] = res
            return cpy if len(cpy) > 0 else None
        except IndexError:
            return deepcopy(d)

    # more than one layer remaining, currently at a mapping / dict
    # replace with result of a recursive call
    elif isinstance(d, collections.Mapping):
        try:
            res = remove_path_from_dict(d[first_filter], sep.join(filter_paths[1:]))
            cpy = deepcopy(d)
            if res is None:
                del cpy[first_filter]
            else:
                cpy[first_filter] = res
            return cpy if len(cpy) > 0 else None
        except KeyError:
            return deepcopy(d)

    # intermediate element is neither sequence nor mapping, just discard
    else:
        return None


def get_path_from_dict(d, path, keep_structure=True, sep='/'):
    """
    get elements from a nested dict based on a XPath-like path
    can handle both nested dicts/mappings and nested lists/sequences
    can return either the value or a nested datastructure like d, but containing only the value at the path
    """

    filter_paths = path.strip(sep).split(sep)
    first_filter = filter_paths[0]

    # end of recursion at element, return
    if first_filter == '':
        return d

    # end of recursion at list or dict, return value at index (optionally wrap again if keep_structure)
    if len(filter_paths) == 1:
        if isinstance(d, collections.Sequence) and first_filter.isdigit():
            try:
                return [d[int(first_filter)]] if keep_structure else d[int(first_filter)]
            except IndexError:
                return None
        elif isinstance(d, collections.Mapping):
            try:
                return {first_filter: d[first_filter]} if keep_structure else d[first_filter]
            except KeyError:
                return None
        else:
            return None

    # intermediate nesting: return result of recursive call, optionally wrapped
    if isinstance(d, collections.Sequence) and first_filter.isdigit():
        try:
            res = get_path_from_dict(d[int(first_filter)], sep.join(filter_paths[1:]), keep_structure)
            if res is None:
                return None
            return [res] if keep_structure else res
        except IndexError:
            return None
    elif isinstance(d, collections.Mapping):
        try:
            res = get_path_from_dict(d[first_filter], sep.join(filter_paths[1:]), keep_structure)
            if res is None:
                return None
            return {first_filter: res} if keep_structure else res
        except KeyError:
            return None

    # path not finished but no more nested levels
    else:
        return None


def generate_recursive_dict(data, path, sep='/'):
    """
    generate a recursive dict in which a data value is wrapped in multiple layers of dicts,
    given by an XPath-style path
    """

    # list of path levels
    paths = path.strip(sep).split(sep)
    first_path = paths[0]

    # end of recursion, return the value
    if first_path == '':
        return data
    else:
        # add one level to result, then recurse
        return {first_path: generate_recursive_dict(data, sep.join(paths[1:]))}


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
