def clamp(x, min_x, max_x):
    return max(min_x, min(x, max_x))

def middle2corner(ms_middle_coordinates, fov):
    """
    Gives the ability to calculate from the global coordinates of the microscope back to the upper left corner of an
    image
    :param ms_middle_coordinates: current coordinates of the measurement for n dimensions
    :param fov: field of view sizes of n dimensions
    :return: coordinates of the upper left corner for n dimensions
    """
    corner_coords = []
    for i in range(len(ms_middle_coordinates)):
        corner_coords.append(ms_middle_coordinates[i] - (0.5*fov[i]))
    return corner_coords

# TODO call pixel_fov_dimensions -> pixel_size
def corner2spot(corner_coords, fspot_coords, pixel_fov_dimensions):
    """
    Calculates the coordinates of spots after processing with Fiji. Adds the vector coordinates calculated
    by Fiji to the corner coordinates
    :param corner_coords: coordinates of the upper left corner of the fov i.e. (0|0) for Fiji
    :param fspot_coords: coordinates calculated by Fiji
    :return: returns the actual, global coordinates
    """
    actual_gcoords = []
    factor = (pixel_fov_dimensions[0], pixel_fov_dimensions[1])
    for i in range(len(fspot_coords)):
        actual_gcoords.append([(int(corner_coords[0])+(int(fspot_coords[i][0]))*factor[i]),
                              (int(corner_coords[1])+(int(fspot_coords[i][1]))*factor[i])])
    return actual_gcoords




def flatten_dict(d, prefix):
    if isinstance(d, dict):
        dicts = list()
        for (k,v) in d.items():
            dicts.append(flatten_dict(v, "/".join([prefix, k.decode('utf-8')])))
        return reduce(lambda x, y: dict(list(x.items()) + list(y.items())), dicts)
    elif isinstance(d, list):
        dicts = list()
        for i in range(len(d)):
            dicts.append(flatten_dict(d[i], "/".join([prefix, str(i)])))
        return reduce(lambda x, y: dict(list(x.items()) + list(y.items())), dicts)
    else:
        return {prefix: d}

def bench_coords_snapshot(ms):
    x = ms.parameter("OlympusIX/scanrange/x/offset")
    y = ms.parameter("OlympusIX/scanrange/y/offset")
    z = ms.parameter("OlympusIX/scanrange/z/off")
    return [x, y, z]


def fov_len_snapshot(ms):
    x = ms.parameter("ExpControl/scan/range/x/len")
    y = ms.parameter("ExpControl/scan/range/y/len")
    z = ms.parameter("ExpControl/scan/range/z/len")
    return [x, y, z]


def scan_offset_coords_snapshot(ms):
    x = ms.parameter("ExpControl/scan/range/x/off")
    y = ms.parameter("ExpControl/scan/range/y/off")
    z = ms.parameter("ExpControl/scan/range/z/off")
    return [x, y, z]