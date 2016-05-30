# coding UTF-8
from Util.tile_util import middle2corner, corner2spot
import re
#TODO: Funktion um Fiji fov dimensionen zu übergeben + weitergeben an spätere berechnunge am besten innerhalb des Fiji scripts
# start Fiji script for coordinate detection


def call_fiji(path):
    import subprocess
    return subprocess.getoutput([path])

call_fiji('./call-fiji-bashscript')


def read_coords(path):
    """
    Code to read data from the temp file i.e. coordinates for registered spots
    :param path: path in form of  "/path/where/the/txt/is"
    :return: coordsffs: coordinates of the spots in form of a list i.e. [(X1,Y1),(X2,Y2)...] for n coords
    """
    file = open(path, 'r')
    file = file.read()
    items = re.findall('[0-9].[0-9]+', file)
    coordsffs = []
    xitems = items[0::2]
    yitems = items[1::2]
    for i in range(len(xitems)):
        coordsffs.append((xitems[i], (yitems[i])))
    return coordsffs


print(read_coords("coords-temp"))

def find_hssites(fiji_script_path, coords_temp_path, fov_dimensions, global_coordinates):
    sites = call_fiji(fiji_script_path)
    sites = read_coords(coords_temp_path)
    # TODO Umrechnungsfaktor einbauen für mm
    # dimension independent calculation of corner coordinates
    corner_coords = middle2corner(global_coordinates, fov_dimensions)
    return corner2spot(corner_coords, sites)
