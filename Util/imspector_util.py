def get_fov_dimensions(ms):
    """
    :param ms: Measurement
    :return: Tuple containing length's of the field of view
    """
    return ms.parameter("ExpControl/scan/range/x/len"), ms.parameter("ExpControl/scan/range/y/len")


def acquire_measurement_at_coordinates(im, ms, coords):
    amount_of_measurements = len(coords)
    for i in range(amount_of_measurements):
        amove(ms, coords[i][0], coords[i][1])
        print(coords[i])
        im.run(ms)
        for ii in range(ms.active_configuration().number_of_stacks()):
            outfd.write(ms.active_configuration().stack(ii))
        a = input("Enter for continue, or type something to stop: ")
        if a != "":
            break
    outfd = None
    return None

"""# not yet perfect. Just for testing
def measurement():
    filename = "C:\\Users\\RESOLFT\\Desktop\\Tiled\\20160426_tiling02.msr"
    outfd = File(filename, File.Write)
    coords = generate_grid_snake((0,0),(2e-4 , 2e-4),get_fov_dimensions(ms), overlap = 0.1)
    acquire_measurement_at_coordinates(im, ms, coords)
"""

def amove_calc(x2, y2):
    """
    Calculates the coordinates for the absolute movement. "amove" for absolute move
    :param x2: integer. Absolute position for bench in x direction (in mm)
    :param y2: integer. Absolute position for bench in y direction (in mm)
    :return: 2 integers: x,y; returns the absolute values for x and y in mm if both are smaller than their Max's
    :return: None: if x or y are bigger than their Max's
    """
    if -59*1e-3 < x2 < 59*1e-3 and -38*1e-3 < y2 < 38*1e-3:
        # The values are getting rounded so that values < 1e-7 don't confuse the microscopes settings
        x = round(x2, 7)
        y = round(y2, 7)
        return x, y
    elif (-59*1e-3 < x2 < 59*1e-3) is False:
        print('X Coordinate is out of range. Can not move this far')
    elif (-38*1e-3 < y2 < 38*1e-3) is False:
        print('Y Coordinate is out of range. Can not move this far')
    else:
        print('Unexpected Error')


def move_x_calc(x2):
    """
    :param x2: Integer: Value by which the bench shall be move in x direction (in mm)
    :return: integer: If the Value is smaller than it's Max
    :return: None: If the Value is bigger than it's Max
    """
    x = ms.parameter("OlympusIX/scanrange/x/offset")
    if -59*1e-3 < (x2 + x) < 59*1e-3:
        x += x2
        return x
    elif not (-59*1e-3 < (x2 + x) < 59*1e-3):
        print('X Coordinate is out of range. Can not move this far')


def move_y_calc(y2):
    """
    Makes sure that the value by which the move should be
    :param y2: Integer: Value by which the bench shall be move in x direction (in mm)
    :return: integer: If the Value is smaller than it's Max
    :return: None: If the Value is bigger than it's Max
    """
    y = ms.parameter("OlympusIX/scanrange/y/offset")
    if -38*1e-3 < (y2 + y) < 38*1e-3:
        y += y2
        return y
    elif not (-38*1e-3 < (y2 + y) < 38*1e-3):
        print('Y Coordinate is out of range. Can not move this far')


def amove(ms, x2, y2):
    """
    loads the coordinates for the movement and changes the values for the local coordinates in the config file
    :param x2: integer: absolute x coordinate
    :param y2: integer: absolute y coordinate
    :return: moves the bench to the absolute coordinates if the conditions of move_absolute_calc are True
    """
    coordinates = amove_calc(x2, y2)
    # Epsilon is needed to set parameters correctly ending with a 0
    eps = 1e-8
    x_target_coordinate = float(coordinates[0] if coordinates[0] == 0 else coordinates[0] + eps)
    y_target_coordinate = float(coordinates[1] if coordinates[1] == 0 else coordinates[1] + eps)
    if x_target_coordinate is not None and y_target_coordinate is not None:
        ms.set_parameter("OlympusIX/scanrange/x/offset", x_target_coordinate)
        ms.set_parameter("OlympusIX/scanrange/y/offset", y_target_coordinate)


def move_x(ms, x2):
    """
    Moves the bench relative to the x axis
    :param x2: integer: relative x coordinate provided by move_x_calc
    :return: move the bench x mm
    """
    move_by_x = move_x_calc(x2)
    if move_by_x is not None:
        ms.set_parameter("OlympusIX/scanrange/x/offset", move_by_x)


def move_y(ms, y2):
    """
    Moves the bench relative to the y axis
    :param x2: integer: relative y coordinate provided by move_y_calc
    :return: move the bench y mm
    """
    move_by_y = move_y_calc(y2)
    if move_by_y is not None:
        ms.set_parameter("OlympusIX/scanrange/y/offset", move_by_y)


def move(ms, x2, y2):
    """
    moves the bench relative in x,y direction (mm)
    :param x2:  integer: relative x coordinate provided by move_x_calc
    :param y2:  integer: relative y coordinate provided by move_y_calc
    :return: moves the bench x,y mm in x,y direction
    """
    move_by_x = move_x_calc(x2)
    move_by_y = move_y_calc(y2)
    if move_by_x is not None and move_by_y is not None:
        ms.set_parameter("OlympusIX/scanrange/x/offset", move_by_x)
        # time.sleep(1)
        ms.set_parameter("OlympusIX/scanrange/y/offset", move_by_y)