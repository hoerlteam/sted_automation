import csv
import time
from sys import stdout

def setup_csv_writer(fd):
    fieldnames = ['file', 'type', 'time_complete', 'stg_x', 'stg_y', 'stg_z', 'scan_x', 'scan_y', 'scan_z']
    writer = csv.DictWriter(fd, fieldnames=fieldnames, extrasaction='ignore')
    writer.writeheader()
    return writer

def make_csv_line(filename, t = None, stg_coords = [0,0,0], scan_coords=[0,0,0], type='confocal'):

    if t == None:
        t = time.time()

    res = dict()
    res['file'] = filename
    res['type'] = type
    res['time_complete'] = t
    res['stg_x'] = stg_coords[0]
    res['stg_y'] = stg_coords[1]
    res['stg_z'] = stg_coords[2]
    res['scan_x'] = scan_coords[0]
    res['scan_y'] = scan_coords[1]
    res['scan_z'] = scan_coords[2]

    return res

def main():
    w = setup_csv_writer(stdout)

    for i in range(5):
        l = make_csv_line(i)
        w.writerow(l)
        time.sleep(1)

if __name__ == '__main__':
    main()