#from PIL import Image
from functools import reduce
from .imspector_util import get_fov_dimensions
from .coordinate_util import *
from .coordinates import *
import time


def generate_grid(area_min, area_max, fov_dimensions, overlap=0):
    """
    get a list of coordinates of fields of view (x,y of upper left corner) covering the area specified by area_min and area_max
    :param area_min: list of area minimum coordinates (x0, y0)
    :param area_max: list of area maximum coordinates (x1, y1)
    :param fov_dimensions: list, dimensions of field of view (width, height)
    :param overlap: 0-0.5: fraction of overlap of tiles, will be clamped to that range
    :return: coordinates: list of tuples, containing coordinates
    """

    if overlap < 0 or overlap > 0.5:
        raise Exception("ERROR: Overlap should be between 0 and 0.5")

    """
    Legend for variables:
    x0 = area_min[0]
    y0 = area_min[1]
    x1 = area_max[0]
    y1 = area_max[1]
    width = fov_dimensions[0]
    height = fov_dimensions[1]
    """

    # width height of sight
    dx = area_max[0] - area_min[0]
    dy = area_max[1] - area_min[1]
    coordinates = []
    i = 0
    ii = 0

    # Calculating coordinates:
    while i <= dy:
        while ii <= dx:
            coordinate = (ii, i)
            coordinates.append(coordinate)
            ii += (1-overlap)*fov_dimensions[0]
        i += (1-overlap)*fov_dimensions[1]
        ii = 0
    return coordinates


def generate_grid_snake(area_min, area_max, fov_dimensions, overlap=0):
    """
    get a list of coordinates of fields of view (x,y of upper left corner) covering the area specified by area_min and
    area_max. To minimize mechanical wear this function returns the order of coordinates in 'snake' form.
    :param area_min: list of area minimum coordinates (x0, y0)
    :param area_max: list of area maximum coordinates (x1, y1)
    :param fov_dimensions: list, dimensions of field of view (width, height)
    :param overlap: 0-0.5: fraction of overlap of tiles, will be clamped to that range
    :return: coordinates: list of tuples, containing coordinates
    """
    if overlap < 0 or overlap > 0.5:
        raise Exception("ERROR: Overlap should be between 0 and 0.5")

    # Legend for variables:
    x0 = area_min[0]
    y0 = area_min[1]
    x1 = area_max[0]
    y1 = area_max[1]
    width = fov_dimensions[0] * 1-overlap
    height = fov_dimensions[1] * 1-overlap

    # width height of sight
    dx = area_max[0] - area_min[0]
    dy = area_max[1] - area_min[1]

    # List of x's (left to right)
    x_forward_list = [area_min[0]]
    cuts = dx//width
    i = 0
    while i < cuts:
        i += 1
        x_forward_list.append(area_min[0] + (i * width))

    # List of y's (all)
    y_forward_list = [area_min[1]]
    cuts = dy//height
    i = 0
    while i < cuts:
        i += 1
        y_forward_list.append(area_min[0] + (i * height))

    # Backward list of x's (right to left)
    x_backward_list = []
    for i in x_forward_list:
        x_backward_list.append(i)
    x_backward_list.sort(reverse=True)

    # y lists. One explicit list for the movement from left to right and on for the movement from left to right
    y_odd = y_forward_list[0::2]
    y_even = y_forward_list[1::2]

    # Actual grid comes here
    # how_often displays how often to do the "double snake" in order to cover the whole grid
    grid_coordinates = []
    how_often = len(y_forward_list)//2
    for ii in range(how_often):
        for x in x_forward_list:
            grid_coordinates.append((x, y_odd[ii]))
        for x in x_backward_list:
            grid_coordinates.append((x, y_even[ii]))
        ii += 1
    if len(y_forward_list) % 2 != 0:
        for x in x_forward_list:
            grid_coordinates.append((x, y_odd[-1]))
    return grid_coordinates


#  -> maybe make them menber methods of the Coordinates class
def generate_grid_oop(area_min, area_max, fov_dimensions, overlap=0):
    grid_coords = generate_grid_snake(area_min, area_max, fov_dimensions, overlap)
    list_of_coords_objects = []
    for i in range(len(grid_coords)):
        co = Coordinates()
        # Hacky way to get around 2d here, beware!!!
        # old: co.set_bench_coords(grid_coords[i])
        co.set_bench_coords(grid_coords[i] + tuple([0]))
        list_of_coords_objects.append(co)
    return list_of_coords_objects


class EstimatedTime:
    def __init__(self, n):
        self.amount_o_m = n
        self.times = []
        self.timer = [0, 0]

    def print_estimated_time(self):
        """
        Use this function, when the estimated time shall be printed
        :return:
        """
        timer = 0
        for i in self.times:
            timer += sum(i)/len(i)
        timer = timer*self.amount_o_m
        print("Estimated time left:" + (str(timer/60)) + "minutes")

    def add_measure_type(self):
        self.times.append([])

    def start_time(self):
        """
        Use this method before recording an image
        :return:
        """
        self.timer[0] = time.time()

    def stop_time(self, m_type=0):
        """
        This function after image is recorded
        :param m_type: Define what kind of measurement it is
        :return:
        """
        self.timer[1] = time.time()
        self.times[m_type].append(self.timer[1]-self.timer[0])
        self.timer[0], self.timer[1] = 0, 0


def ggacp(ms, fov, n):
    """
    "Generate Grid Around Current Position"
    This function takes the current fov of view that is displayed in Imspector and acquires images
    around this field of view. It can be used for searching for interesting points manually and then
    taking images around the interesting area.

    Example:
    n = 0    n = 1             n = 2               n
                             _ _ _ _ _
             _ _ _          |_|_|_|_|_|
     _      |_|_|_|         |_|_|_|_|_|
    |_|  -> |_|_|_|   ->    |_|_|_|_|_|    ->     ...
            |_|_|_|         |_|_|_|_|_|
                            |_|_|_|_|_|


    n is the amount of "circles" around the image of interest.
    The requested are then gets calculated and a grid is generated using the generate_grid_oop function
    :param ms: measurement object
    :param fov: field of view for image acquisition afterwards
    :param n: Amount of rows
    :return: List of coordinates for image acquisition
    """

    bench = ms.parameter()  # parameter for bench
    offset = ms.parameter()  # parameter for offset
    gbench = ms.parameter()  # parameter for global coords?
    global_coordinates = [(bench[0] + offset[0] + gbench[0]), (bench[1] + offset[1] + gbench[1])]

    # Corner coordinates of the current field of view in Imspector
    corner = middle2corner(global_coordinates, fov)

    # Calculating the upper left corner of the field, depending on the amount of rows
    area_min = [(corner[0] - (fov[0]*n)), (corner[1] - (fov[1]*n))]

    # Calculating the bottom right corner of the filed, depending on the amount of rows
    area_max = [(corner[0]+(fov[0]*((n*2)+1))), (corner[1]+(fov[1]*((n*2)+1)))]

    list_of_coords = generate_grid_oop(area_min, area_max, fov)
    return list_of_coords


def df_circle_generator(fov):
    """
    "David's fancy circle" generator:

    Usage example:
    for i in df_circle:
        acquire_measurement(i)

    Generates a circle of infinite coordinates relative to the center coordinates.
                     ...
                      ^
                       \\
                        O  ->O  ->O  ->O ->O
                         ^                 |
                          \\               v
                        O   O-> O->  O     O
                        ^    ^       |     |
                        |     \\     v     v
                        O   O   O    O     O
                        ^   ^        |     |
                        |   |        v     v
                        O   O <-O<-  O     O
                        ^                  |
                        |                  v
                        O <-O   <-O <- O<- O


    :param fov: [x,y] length of the fields of view
    :return: returns relative position to center (global coordinates)
    """
    n = 0
    corners = [0, 0, 0, 0]
    co = Coordinates()
    co.set_bench_coords([0, 0, 0])
    yield co
    while True:
        bookmark = [0, 0]
        bookmark[0] += corners[2]
        bookmark[1] += corners[3]
        while bookmark[0] < corners[0]:
            co.set_bench_coords(bookmark)
            yield co
            bookmark[0] += fov[0]
        while bookmark[1] > corners[1]:
            co.set_bench_coords(bookmark)
            yield co
            bookmark[1] -= fov[1]
        while bookmark[0] > corners[2]:
            co.set_bench_coords(bookmark)
            yield co
            bookmark[0] -= fov[0]
        while bookmark[1] < corners[3]:
            co.set_bench_coords(bookmark)
            yield co
            bookmark[1] += fov[1]
        n += 1
        corners = [fov[0]*n, fov[1]*(-n), fov[0]*(-n), fov[1]*n]
