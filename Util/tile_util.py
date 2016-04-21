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

def generate_grid_snake(area_min, area_max, fov_dimensions, overlap = 0):
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
        # move right
        if ii < area_min[0]:
            ii += (1-overlap)*fov_dimensions[0]
            while ii <= dx:
                coordinate = (ii, i)
                coordinates.append(coordinate)
                ii += (1-overlap)*fov_dimensions[0]
        elif ii >= area_min[0]:
            while ii <= dx:
                coordinate = (ii, i)
                coordinates.append(coordinate)
                ii += (1-overlap)*fov_dimensions[0]
        else:
            raise Exception("ERROR: Unexpected Error while calculating new coordinate")
        i += (1-overlap)*fov_dimensions[1]
        if not i > dy:
            if ii > dx:
                ii -= (1-overlap)*fov_dimensions[0]
                while ii >= area_min[0]:
                    coordinate = (ii, i)
                    coordinates.append(coordinate)
                    ii -= (1-overlap)*fov_dimensions[0]
            elif ii <= dx:
                while ii >= area_min[0]:
                    coordinate = (ii, i)
                    coordinates.append(coordinate)
                    ii -= (1-overlap)*fov_dimensions[0]
            else:
                raise Exception("ERROR: Unexpected Error while calculating new coordinate")
        else:
            return coordinates
        i += (1-overlap)*fov_dimensions[1]
    return coordinates
