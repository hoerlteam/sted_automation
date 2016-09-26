import re, os, sys, itertools
from ij.io import DirectoryChooser, OpenDialog
from ij import IJ, ImagePlus
from loci.plugins import BF

# manually import ImporterOptions, as the package name contains the "in" constant
ImporterOptions = __import__("loci.plugins.in.ImporterOptions", globals(), locals(), ['object'], -1)

def importMSR(path):
    '''
    open MSR files
    returns array of stacks
    '''

    try:
        io = ImporterOptions()
        io.setId(path)
        io.setOpenAllSeries(True)
        imps = BF.openImagePlus(io)
    except ImagePlus:
        IJ.log("ERROR while opening image file " + path)

    return (imps)

def main():
	dc = DirectoryChooser("Choose directory to process!")
	inputDir = dc.getDirectory()

	if not inputDir:
		return

	# create output directory
	if not os.path.exists(os.path.join(inputDir, "tiffs")):
		os.makedirs(os.path.join(inputDir, "tiffs"));
	
	files = [f for f in os.walk(inputDir).next()[2] if f.endswith('.msr')]

	for f in files:
		imps = importMSR(os.path.join(inputDir, f))

		for i in range(len(imps)):
			IJ.saveAsTiff(imps[i], os.path.join(inputDir, 'tiffs', f + 'ch' + str(i) + '.tif'))

	print('-- ALL DONE --')

if __name__ == '__main__':
	main()