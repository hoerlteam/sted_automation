from Util import tile_util

def main():
    min_coords = [0, 0]
    max_coords = [25, 10]
    fov_size = [5, 5]

    print(tile_util.generate_grid_snake(min_coords, max_coords, fov_size, 0.1))

if __name__ == '__main__':
    main()

