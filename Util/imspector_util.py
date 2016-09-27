"""#import numpy as np
"""
try:
    from specpy import *
except ImportError:
    print("WARNING: no specpy found")
"""
im = Imspector()
ms = im.active_measurement()
params = ms.parameters()
"""
import skimage
import skimage.data
import numpy as np
import matplotlib.pyplot as plt
from Util.tile_util import generate_grid_snake
import time
from Util import datastructures


def get_fov_dimensions(ms):
    """
    :param ms: Measurement
    :return: Tuple containing length's of the field of view
    """
    return ms.parameter("ExpControl/scan/range/x/len"), ms.parameter("ExpControl/scan/range/y/len"), ms.parameter("ExpControl/scan/range/z/len")

def get_pixel_size(ms):
    return ms.parameter("ExpControl/scan/range/x/psz"), ms.parameter("ExpControl/scan/range/y/psz"), ms.parameter("ExpControl/scan/range/z/psz")

def get_fov_dims_pixel(ms):
    """
    Extracts information about the relation of pixel/m out of the ms-object
    :param ms: ms-object
    :return: returns pixel per m
    """
    return ms.parameter("ExpControl/scan/range/x/res"), ms.parameter("ExpControl/scan/range/y/res"), ms.parameter("ExpControl/scan/range/z/res")


# TODO: Das hier muss alles zu einer klasse werden
def amove_calc(x2, y2):
    """
    Calculates the coordinates for the absolute movement. "amove" for absolute move
    :param x2: integer. Absolute position for bench in x direction (in mm)
    :param y2: integer. Absolute position for bench in y direction (in mm)
    :return: 2 integers: x,y; returns the absolute values for x and y in mm if both are smaller than their Max's
    :return: None: if x or y are bigger than their Max's
    """
    if -59 * 1e-3 < x2 < 59 * 1e-3 and -38 * 1e-3 < y2 < 38 * 1e-3:
        # The values are getting rounded so that values < 1e-7 don't confuse the microscopes settings
        x = round(x2, 7)
        y = round(y2, 7)
        return x, y
    elif (-59 * 1e-3 < x2 < 59 * 1e-3) is False:
        raise Exception("ERROR: X Coordinate is out of range. Can not move this far")
    elif (-38 * 1e-3 < y2 < 38 * 1e-3) is False:
        raise Exception("ERROR: X Coordinate is out of range. Can not move this far")
    else:
        raise Exception("ERROR: Y Coordinate is out of range. Can not move this far")


def move_x_calc(x2):
    """
    :param x2: Integer: Value by which the bench shall be move in x direction (in mm)
    :return: integer: If the Value is smaller than it's Max
    :return: None: If the Value is bigger than it's Max
    """
    x = ms.parameter("OlympusIX/scanrange/x/offset")
    if -59 * 1e-3 < (x2 + x) < 59 * 1e-3:
        x += x2
        return x
    elif not (-59 * 1e-3 < (x2 + x) < 59 * 1e-3):
        raise Exception("ERROR: X Coordinate is out of range. Can not move this far")


def move_y_calc(y2):
    """
    Makes sure that the value by which the move should be
    :param y2: Integer: Value by which the bench shall be move in x direction (in mm)
    :return: integer: If the Value is smaller than it's Max
    :return: None: If the Value is bigger than it's Max
    """
    y = ms.parameter("OlympusIX/scanrange/y/offset")
    if -38 * 1e-3 < (y2 + y) < 38 * 1e-3:
        y += y2
        return y
    elif not (-38 * 1e-3 < (y2 + y) < 38 * 1e-3):
        raise Exception("ERROR: Y Coordinate is out of range. Can not move this far")


def amove(ms, x2, y2):
    """
    loads the coordinates for the movement and changes the values for the local coordinates in the config file
    :param ms: object: active_measurement "object"
    :param x2: integer: absolute x coordinate
    :param y2: integer: absolute y coordinate
    :return: moves the bench to the absolute coordinates if the conditions of move_absolute_calc are True
    """
    coordinates = amove_calc(x2, y2)
    # Epsilon is needed to set parameters correctly ending with a 0
    eps = 1e-8
    x_target_coordinate = float(coordinates[0] if coordinates[0] == 0 else coordinates[0] + eps)
    y_target_coordinate = float(coordinates[1] if coordinates[1] == 0 else coordinates[1] + eps)
    if x_target_coordinate is not None and y_target_coordinate is not None:
        # TODO: Syntax ändern hier nach Settings objekt
        ms.set_parameter("OlympusIX/scanrange/x/offset", x_target_coordinate)
        ms.set_parameter("OlympusIX/scanrange/y/offset", y_target_coordinate)


def move_x(ms, x2):
    """
    Moves the bench relative to the x axis
    :param ms: ms: object: active_measurement "object"
    :param x2: integer: relative x coordinate provided by move_x_calc
    :return: move the bench x mm
    """
    move_by_x = move_x_calc(x2)
    if move_by_x is not None:
        ms.set_parameter("OlympusIX/scanrange/x/offset", move_by_x)


def move_y(ms, y2):
    """
    Moves the bench relative to the y axis
    :param ms: object: active_measurement "object"
    :param y2: integer: relative y coordinate provided by move_y_calc
    :return: move the bench y mm
    """
    move_by_y = move_y_calc(y2)
    if move_by_y is not None:
        ms.set_parameter("OlympusIX/scanrange/y/offset", move_by_y)


def move(ms, x2, y2):
    """
    moves the bench relative in x,y direction (mm)
    :param ms: object: active_measurement "object"
    :param x2:  integer: relative x coordinate provided by move_x_calc
    :param y2:  integer: relative y coordinate provided by move_y_calc
    :return: moves the bench x,y mm in x,y direction
    """
    move_by_x = move_x_calc(x2)
    move_by_y = move_y_calc(y2)
    if move_by_x is not None and move_by_y is not None:
        ms.set_parameter("OlympusIX/scanrange/x/offset", move_by_x)
        # time.sleep(1)
        ms.set_parameter("OlympusIX/scanrange/y/offset", move_by_y)


def acquire_measurement_at_coordinates(im, ms, l_of_coords, configs_path, out_path):
    conf = Settings()
    if type(configs_path) != str:
        raise Exception("ERROR: configs parameter must be str!")
    conf.load_from_file(configs_path)
    #TODO: passt das so?
    conf.apply_to_settings_dict(params)
    amount_of_measurements = len(l_of_coords)
    name = generate_random_name2()
    for i in range(amount_of_measurements):
        amove(ms, l_of_coords[i][0], l_of_coords[i][1])
        # print(l_of_coords[i])
        im.run(ms)
        save_stack(out_path, name, i)
        a = input("Enter for continue, or type something to stop: ")
        if a != "":
            break
    return None


def acquire_measurement3(im, ms, configs_path, out_path, name, salt):
    conf = datastructures.Settings()
    if not isinstance(configs_path, str):
        raise Exception("ERROR: configs parameter must be str!")
    conf.load_from_file(configs_path)
    params = ms.parameters()
    conf.apply_to_settings_dict(params)
    ms.set_parameters(params)
    im.run(ms)
    save_stack(ms, out_path, name, salt)


def acquire_measurement(im, config):
    '''
    Wrapper for acquire_measurement_dummy. For aesthetic reasons.
    '''
    acquire_measurement_dummy(im, config)


def acquire_measurement_dummy(im, config, justmove=False, delay=0.5):
    '''
    test function to acquire a measurement with settings, does not return/save anything
    '''
    im.create_measurement()
    
    time.sleep(delay)
    
    ms = im.active_measurement()
    params = ms.parameters()
    config.apply_to_settings_dict(params)
    ms.set_parameters(params)
    
    time.sleep(delay)
    
    if not justmove:
        im.run(ms)
    # here are changes the save stack was not there before:
    #save_stack(ms, 'C://Users//RESOLFT//Desktop//TEST_folder//', '28_06', i)


# TODO: Redo this if David likes the idea of my name class - Set name object instead of the 3 varaibles
def generate_file_for_measurement(path, name, salt=""):
    if not isinstance(path, str):
        raise Exception("ERROR: path must be str!")
    filename = str(path) + str(name) + str(salt)
    # File comes from specpy
    outfd = File(filename, File.Write)
    return outfd


def save_stack(ms, path, name, i):
    """
    Saves the stack. A random name should be generated before plus in order to many images, the counter of which image
    is taken should be given as i.
    :param name: needs a name from the random name generator
    :param i: current number of the measurement
    :return: None. Saves the image into a file
    """
    outfd = generate_file_for_measurement(path, name, i)
    for ii in range(ms.active_configuration().number_of_stacks()):
        outfd.write(ms.active_configuration().stack(ii))
    outfd = None


def config_magic(path):
    """
    Cuts a path into pieces and generates a string for for the set_parameters function
    :param path: config file path
    :return: path cut to fit in the set_parameters syntax i.e.: [b"xy"][..]..
    """
    l = []
    # cutting the path into parts of a list:
    for i in path.split('/'):
        l.append(i)
    # building the syntax:
    a = ""
    for i in range(len(l)):
        if l[i].isdigit():
            a += str('[') + str(l[i]) + str(']')
        else:
            a += str('[b"') + str(l[i]) + str('"]')
    # syntax for setting parameters is params[b"xy"][..].. = z. For that the paths are reframed here
    #print(a)
    return a


def changing_config(ms, params):
    """
    Old function that served as template for the config magic syntax - Delete if not needed
    :param ms:
    :param params:
    :return:
    """
    # deleting some values in order not to crash
    params.pop(b"is_active")
    params.pop(b"prop_driver")
    params.pop(b"prop_version")

    # params must be given in this form or must be automatically translated from path to this form via "config_magic"
    print(params[b"ExpControl"][b"scan"][b"range"][b"x"][b"len"])
    params[b"OlympusIX"][b"scanrange"][b"y"][b"offset"] = -1e-02
    params[b"OlympusIX"][b"scanrange"][b"x"][b"offset"] = -1e-02
    ms.set_parameters(params)


def generate_random_name():
    """
    Generates random name with adler (short name)
    :return: random name as string
    """
    import zlib
    return str(zlib.adler32(bytes(str(time.time() * 1000), "utf-8")))


def generate_random_name2():
    """
    Generates a random name for saving images. Md5 is used. Longer names
    :return: str: name as string
    """
    import hashlib
    hash_object = hashlib.md5(bytes(str(time.time() * 1000), "utf-8"))
    hex_dig = hash_object.hexdigest()
    return str(hex_dig)


def autofocus(im, ms, series=0, dimension=2):
    im.run()
    img = ms.stack(series).data()[0, 0, :, :]

    # dimensions of image
    sh = np.shape(img)

    # define dimension here if needed
    # dimension = 1

    normvar = []
    for i in range(sh[dimension]):
        # get slice i

        # tImg = img[:,:,i]

        if dimension == 0:
            tImg = img[i, :, :]
        elif dimension == 1:
            tImg = img[:, i, :]
        elif dimension == 2:
            tImg = img[:, :, i]

    # edge-filter slice
    tImg = skimage.filters.sobel(tImg)
    # calculate var/mean
    normvar.append(np.var(tImg) / np.mean(tImg))

    # manual output:
    # plt.plot(normvar)
    # print maximum index
    # print(normvar.index(max(normvar)))

    return normvar.index(max(normvar))

