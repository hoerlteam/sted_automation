import random
import numpy as np


def fiji_dummy(fov):
    """
    Random function that mimics the behaviour of the Fiji_Dot detection function in order to test the
    functionality of the rest of the independent of the actual samples
    :param fov: list/tuple of 2 parameters that mimics the behaviour of the real fov
    :return:
    """
    xfov = fov[0]
    yfov = fov[1]
    fspots = []
    a = random.randint(1, 5)
    for i in range(a):
        ffs1 = ((random.randint(1, 100))/100)
        ffs2 = ((random.randint(1, 100))/100)
        b = ([((xfov-random.randint(0, xfov)) + ffs1), ((yfov-random.randint(0, yfov))+ ffs2)])
        fspots.append(b)
    return fspots


def save_coords_to_temp(coords, tfile="coords-temp"):
    file = open(str(tfile), 'w')
    file.write(str(coords))
    file.close()


def do_dummy_stuff(fov, outfile):
    a = fiji_dummy(fov)
    save_coords_to_temp(a, outfile)
