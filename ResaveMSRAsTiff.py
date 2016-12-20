# @String path
# @boolean(value=false) allseries

import sys, os

sys.path.append(os.getcwd())
import ij_io

from ij.io import DirectoryChooser, OpenDialog

def main():
    if path == None:
        dc = DirectoryChooser("Choose directory to process!")
        inputDir = dc.getDirectory()

        if not inputDir:
            return
    else:
        inputDir = path

    ij_io.resave_msr_folder_as_tiff(inputDir, allseries)

if __name__ == '__main__':
    main()