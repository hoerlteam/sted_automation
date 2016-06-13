#! coding:UTF-8
from Util.tile_util import generate_grid_snake
from Util.imspector_util import *
from Util.dot_detection.Fiji_coordinate_detection import find_hssites
Fiji_path = ' ' #hardcoded?
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


def sps(configs, image_path,  macro_path, coordinates, fov_dimensions, overlap=0):
    """
    singe measurement plus STED measurement
    :param configs:
    :param image_path:
    :param macro_path:
    :param coordinates:
    :param fov_dimensions:
    :param overlap:
    :return:
    """
    # implement counter for configs? i=0 -> after measurement i+=1 so that configs[1] gets loaded
    #from specpy import *
    im = Imspector()
    ms = im.active_measurement()
    acquire_measurement_at_coordinates(im, ms, coordinates, configs, image_path)#TODO: add configs here)
    list_of_sites = find_hssites(Fiji_path, macro_path, image_path, fov_dimensions, coordinates)
    acquire_measurement_at_coordinates(im, ms, list_of_sites, configs, image_path)

