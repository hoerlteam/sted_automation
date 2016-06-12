#! coding:UTF-8
from Util.tile_util import generate_grid_snake
from Util.imspector_util import *
from Util.dot_detection import Fiji_coordinate_detection
#
# def sgbs_mesurement(area_min, area_max, fov_dimensions=0 , overlap=0):
#     """
#     "snake grid before STED measurement" takes measurements of the whole grid, stiches the grid and takes
#     STED measuremnts afterwards
#     :param fov_dimensions:
#     :return:
#     """
#     #from specpy import *
#     im = Imspector()
#     ms = im.active_measurement()
#     #params = ms.parameters()
#     #TODO: implementd possiblity to change the fov_dimensions from here
#     fov_dimensions = get_fov_dimensions(ms)
#     grid_coordinates = generate_grid_snake(area_min, area_max, fov_dimensions, overlap)
#     acquire_measurement_at_coordinates(im, ms, grid_coordinates)
#     # TODO: Do stitching here


def smps(path, coordinates, fov_dimensions, overlap=0):
    """
    Single measurement plus STED
    :param area_min:
    :param area_max:
    :param fov_dimensions:
    :param overlap:
    :return:
    """
    from specpy import *
    im = Imspector()
    ms = im.active_measurement()
    acquire_measurement_at_coordinates(im, ms, coordinates, configs, path)#TODO: add configs here)
    Fiji_coordinate_detection.call_fiji()
    settings
    acquire measurements
    peace out
