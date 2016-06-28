import numpy as np
from .coordinate_util import *


class Coordinates:
    """
    This class writes th parameters for the global bench coordinates, the lenght of the field of view
    and the scan offset in a list. [bench_coordinates, fov_lenght, scan_offset].
    """
    def __init__(self, bench_coords=(0, 0, 0), fov_len=(0, 0, 0), offset_coords=(0, 0, 0)):
        self.coordinates = [bench_coords, fov_len, offset_coords]

    def __str__(self):
        return str(self.coordinates)

    def set_bench_coords(self, bench_coords):
        self.coordinates[0] = ensure_nd(bench_coords, 3)
        # TODO do this in other setters

    def set_offset_coords(self, offset_coords):
        self.coordinates[2] = ensure_nd(offset_coords, 3)

    def set_fov_len(self, fov_len):
        self.coordinates[1] = fov_len

    def copy(self):
        res = Coordinates()
        res.coordinates = self.coordinates.copy()
        return res

    def get_all_in_one_offset(self):
        return list(np.array(self.get_bench_coords()) + np.array(self.get_scan_offset()))

    def create_bench_coordinates(self):
        """
        create a new Coordinates object representing the same subspace, but with just a movement of the stage
        (offset_coords of this object will be 0)
        :return:
        """
        res = self.copy()
        newBenchShift = self.get_all_in_one_offset()
        res.set_bench_coords(newBenchShift)
        res.set_offset_coords([0, 0, 0])
        return res

    def get_bench_coords(self):
        """
        :return: returns the coordinates of the bench in form [x, y, z]
        """
        return self.coordinates[0]

    def get_fov_len(self):
        """
        :return: returns the length of the the fov in form [x, y, z]
        """
        return self.coordinates[1]

    def get_scan_offset(self):
        """
        :return: returns the scan-offset in form of [x, y, z]
        """
        return self.coordinates[2]

    def corner_coords(self):
        return middle2corner(self.get_all_in_one_offset(), self.coordinates[1])

    def corner2spot_from_object(self, fspot_pixel_coords, pixel_size):
        return corner2spot(self.corner_coords(), fspot_pixel_coords, pixel_size)

    def middle2spot(self, fspot_coords):
        return np.array(fspot_coords) - np.array(self.get_all_in_one_offset())



def create_coordinates_from_measurement(self, ms):
    bench_coords = (bench_coords_snapshot(ms))
    fov_len = (fov_len_snapshot(ms))
    offset_coords = (scan_offset_coords_snapshot(ms))
    return Coordinates(bench_coords, fov_len, offset_coords)


def main():
    a = Coordinates([0,0,0], [0,0,0], [0,0,0])
    print(a.get_bench_coords())

if __name__ == '__main__':
    main()