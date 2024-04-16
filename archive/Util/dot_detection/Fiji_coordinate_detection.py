# coding UTF-8
from archive.Util.tile_util import middle2corner, corner2spot
import re
# start Fiji script for coordinate detection
# path =  path to Fiji hard coded?
Fiji_path = '/home/pascal/Apps/Fiji.app/ImageJ-linux64'
Fiji_hss_finder = 'Macro.py'
Fiji_cell_finder = 'Macro_bigcell_finder.py'


def call_fiji(Fiji_path, macro_path, image_path, series=1, size=15, threshold=10):
    """
    Calls Fiji, uses Threshold methods and TrackMate for filtering and isolating interesting spots
    :param Fiji_path: Path where the executable Fiji bin is
    :param macro_path: Path of the macro that  shall be used
    :param image_path: Image that shall be analysed
    :param series: Series of the .msr file
    :param size: size of the spots that are searched
    :param threshold: Threshold value for TrackMate thresholding
    :return: Saves coordinates in coords  temp file
    """
    fiji_params={'path_to_Fiji': str(Fiji_path),
                 'macro_path': str(macro_path),
                 'image_path': str(image_path),
                 'series_number': str(series),
                 'size_of_image': str(size),
                 'thresh': str(threshold)}
    import subprocess
    function_call = '{path_to_Fiji} {macro_path} {image_path} {series_number} {thresh} {size_of_image}'.format(**fiji_params)
    return subprocess.getoutput(function_call)


# alternative to trackmate - trackpy

def read_coords(path):
    """
    Code to read data from the temp file i.e. coordinates for registered spots
    :param path: path in form of  "/path/where/the/txt/is"
    :return: coordsffs: coordinates of the spots in form of a list i.e. [(X1,Y1),(X2,Y2)...] for n coords
    """
    file = open(path, 'r')
    file = file.read()
    items = re.findall('[0-9]+.[0-9]+', file)
    coordsffs = []
    xitems = items[0::2]
    yitems = items[1::2]
    for i in range(len(xitems)):
        coordsffs.append((float(xitems[i]), float(yitems[i])))
    return coordsffs


def find_hssites(fiji_path, macro_path, image_path, fov_dimensions, pixel_fov_dimensions, global_coordinates=(0, 0), coords_temp_path='coords-temp'):
    call_fiji(fiji_path, macro_path, image_path)
    sites = read_coords(coords_temp_path)
    corner_coords = middle2corner(global_coordinates, fov_dimensions)
    spots = corner2spot(corner_coords, sites, pixel_fov_dimensions)
    return spots
