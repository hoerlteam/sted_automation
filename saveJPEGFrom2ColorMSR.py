# @String path
# @boolean(value=false) allseries

import sys, os

sys.path.append(os.getcwd())
import ij_io

def main():
    if path == None:
        dc = DirectoryChooser("Choose directory to process!")
        inputDir = dc.getDirectory()

        if not inputDir:
            return
    else:
        inputDir = path

	print(allseries)
    
    for d, _, _ in os.walk(inputDir):
        ij_io.resave_msr_folder_as_jpeg_sum_projection(d, allseries)


if __name__ == '__main__':
	main()