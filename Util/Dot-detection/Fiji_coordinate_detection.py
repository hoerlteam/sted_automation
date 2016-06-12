# coding UTF-8
from Util.tile_util import middle2corner, corner2spot
import re
# start Fiji script for coordinate detection
# path =  path to Fiji hard coded?
Fiji_hss_finder = '/home/pascal/Apps/Fiji.app/ImageJ-linux64 Macro.py'
Fiji_cell_finder = '/home/pascal/Apps/Fiji.app/ImageJ-linux64 Macro_bigcell_finder.py'


def call_fiji(Fiji_path, image_path, series=1, size=15, threshold=1):
    fiji_params={'path_to_Fiji': str(Fiji_path),
                 'image_path': str(image_path),
                 'series_number': str(series),
                 'thresh': str(threshold)}
    import subprocess
    function_call = '{path_to_Fiji} {image_path} {series_number} {thresh}'.format(**fiji_params)
    return subprocess.getoutput(function_call)


call_fiji(Fiji_hss_finder)
#call_fiji(Fiji_cell_finder)


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
        coordsffs.append((xitems[i], (yitems[i])))
    return coordsffs


print(read_coords("coords-temp"))


def find_hssites(fiji_script_path, image_path, fov_dimensions, global_coordinates, coords_temp_path="coords-temp"):
    call_fiji(fiji_script_path, image_path)
    sites = read_coords(coords_temp_path)
    # TODO Umrechnungsfaktor einbauen für mm
    # dimension independent calculation of corner coordinates
    #TODO: Die global coordinates hier sind vorraussichtlich echt wichtig -> evtl über globales feedbakc von amove regeln
    corner_coords = middle2corner(global_coordinates, fov_dimensions)
    return corner2spot(corner_coords, sites)
