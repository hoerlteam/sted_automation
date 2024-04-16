from PIL import Image
import numpy as np


def get_interval(x1, x2, steps):
        intervals = []
        dx = x2-x1
        intervals.append(x1)
        b = steps-1
        i = 1
        while i <= b:
            a = x1 + (i/b)*dx
            intervals.append(a)
            i += 1
        return intervals




def read_image_stack(path, correct16bit=False):
    """
    :param path: path to .tif file
    :return: stack of layers
    """
    image_file = Image.open(path)
    image_list = list()
    n = 0
    while True:
        (w, h) = image_file.size
        image_list.append(np.array(image_file.getdata()).reshape(h, w))
        n += 1
        try:
            image_file.seek(n)
        except:
            break

    res = np.dstack(image_list)
    if correct16bit:
        res = res - np.iinfo(np.int16).max - 1

    return res
