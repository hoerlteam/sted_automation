def generate_grid(area_min, area_max, fov_dimensions ):
    """
    get a list of coordinates of fields of view (x,y of upper left corner) covering the area specified by area_min and area_max
    :param area_min: list of area minimum coordinates (x0, y0)
    :param area_max: list of area maximum coordinates (x1, y1)
    :param fov_dimensions: list, dimensions of field of view (width, height)
    :return: coordinates: list of tuples, containing coordinates
    """

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
            ii += fov_dimensions[0]
        i += fov_dimensions[1]
        ii = 0
    return coordinates
