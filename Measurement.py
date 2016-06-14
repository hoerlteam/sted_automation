#! coding:UTF-8
from Util.tile_util import generate_grid_snake
from Util.imspector_util import *
from Util.dot_detection.Fiji_coordinate_detection import find_hssites
fiji_path = 'C:\\Users\\RESOLFT\\Desktop\\Fiji.app\\ImageJ-win64.exe' #Fiji path on STED commputer
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


# def sps(config_paths, image_path, macro_path, coordinates, overlap=0):
#     """
#     singe measurement plus STED measurement
#     :param config_paths:
#     :param image_path:
#     :param macro_path:
#     :param coordinates:
#     :param fov_dimensions:
#     :param overlap:
#     :return:
#     """
#     # implement counter for configs? i=0 -> after measurement i+=1 so that configs[1] gets loaded
#     #from specpy import *
#     im = Imspector()
#     ms = im.active_measurement()
#     if len(config_paths) < 2:
#         raise Exception("need moar config")
#     acquire_measurement_at_coordinates(im, ms, coordinates, config_paths[0], image_path)#TODO: add configs here)
#     fov_dimensions = get_fov_dimensions(ms)
#     list_of_sites = find_hssites(Fiji_path, macro_path, image_path, fov_dimensions, coordinates)
#     acquire_measurement_at_coordinates(im, ms, list_of_sites, config_paths[1], image_path)


def sps_version2(configs_path, image_path, macro_path, name, salt):
    if len(configs_path) < 2:
        raise Exception("need moar config")
    acquire_measurement(im, ms, configs_path, image_path, name, salt)
    fov_dimensions = get_fov_dimensions(ms)
    pixel_fov_dimesions = get_pixel_dimensions(ms)
    spot_coordinates = find_hssites(fiji_path, macro_path, image_path, fov_dimensions, pixel_fov_dimesions)
