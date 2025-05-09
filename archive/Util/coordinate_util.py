import sys
from skimage.feature import blob_dog, blob_log, blob_doh
import math


def ensure_nd(vec, n=3, padding=0):
    """
    ensure that list v is of length n, pad with zeros if necessary
    """
    to_add = n - len(vec)
    for _ in range(to_add):
        vec.append(padding)
    return vec

def clamp(x, min_x, max_x):
    return max(min_x, min(x, max_x))

def middle2corner(ms_middle_coordinates, fov, pixelsd=None):
    """
    Gives the ability to calculate from the global coordinates of the microscope back to the upper left corner of an
    image
    :param ms_middle_coordinates: current coordinates of the measurement for n dimensions
    :param fov: field of view sizes of n dimensions
    :param pixelsd : pixel size, may be None, if this is given, we will correct FOV in the same way Imspector does it:
                    actual dim = (floor(specified dim / pixel size) - 1) * pixel size
    :return: coordinates of the upper left corner for n dimensions
    """
    fov_corrected = list(fov)

    if pixelsd is not None:
        for i in range(len(fov_corrected)):
            fov_corrected[i] = (math.floor(fov[i] / pixelsd[i]) - 1) * pixelsd[i]

    corner_coords = []
    for i in range(len(ms_middle_coordinates)):
        corner_coords.append(ms_middle_coordinates[i] - (0.5*fov_corrected[i]))
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
    factor = (pixel_fov_dimensions[0], pixel_fov_dimensions[1], pixel_fov_dimensions[2])
    #print(corner_coords, fspot_coords, factor)
    #sys.stdout.flush()
    for i in range(len(fspot_coords)):
        
        actual_gcoords.append([(float(corner_coords[0])+((float(fspot_coords[i][0]))*factor[0])),
                              (float(corner_coords[1])+((float(fspot_coords[i][1]))*factor[1])),
                              (float(corner_coords[2])+((float(fspot_coords[i][2]))*factor[2]))])
    return actual_gcoords


def return_spot_coords(coordinates_object, spots, pixelsd):
    """
    Wrapper for middel2corner and corner2spot functions
    :param coordinates_object: coordinates object
    :param spots: list of coordinates of interesting spots
    :param pixelsd: pixeldimensions. Faktor for pixel - µm calculation
    :return: actual global coordinates of the spots
    """
    # implement pixelsd here?
    corner = middle2corner(coordinates_object.get_scan_offset(), coordinates_object.get_fov_len())
    actual_coords = corner2spot(corner, spots, pixelsd)
    return actual_coords


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


def find_blobs(ms, series=0):
    """
    function for finding blobs in a image
    :param ms: measurement containing an image
    :param series:
    :return: returns coordinates of the "blobs"
    """
    dta = ms.stack(series).data()[0,0,:,:]
    blbs = blob_log(dta, max_sigma=30, num_sigma=10, threshold=0.02)
    res = []
    for b in blbs:
        res.append([b[1], b[0]])
    return res

