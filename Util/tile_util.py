from PIL import Image
def clamp(x, min_x, max_x):
    return max(min_x, min(x, max_x))

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

# TODO: Klasse mit instant middle2corner bzw global nach stithing middle to corner
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


def corner2spot(corner_coords, fspot_coords, pixel_fov_dimensions, fov_dimensions):
    """
    Calculates the coordinates of spots after processing with Fiji. Adds the vector coordinates calculated
    by Fiji to the corner coordinates
    :param corner_coords: coordinates of the upper left corner of the fov i.e. (0|0) for Fiji
    :param fspot_coords: coordinates calculated by Fiji
    :return: returns the actual, global coordinates
    """
    if len(corner_coords) != len(fspot_coords):
        raise Exception("ERROR: Different amount of dimensions between corner coordinates and coordinates"
                        "provided by Fiji")
    actual_gcoords = []
    factor = calculate_pixel_to_mm(pixel_fov_dimensions, fov_dimensions)
    for i in range(len(corner_coords)):
        actual_gcoords.append(int(corner_coords[i])+(int(fspot_coords[i]))*factor)
    return actual_gcoords


def calculate_pixel_to_mm(pixel_fov_dimensions, fov_dimensions):
    pwidth, pheight = pixel_fov_dimensions[0], pixel_fov_dimensions[1]
    awidth = fov_dimensions[0]
    aheight = fov_dimensions[1]
    if (pwidth/awidth) != (pheight/aheight):
        raise Exception("ERROR: Images did not match. could not calculate factor")
    else:
        return int(pwidth/awidth)


