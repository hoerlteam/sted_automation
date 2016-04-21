from PIL import Image
import numpy as np

def get_interval(start, to, nSteps):




def read_image_stack(path):
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
    return np.dstack(image_list)

