from loci.plugins import BF
# manually import ImporterOptions, as the package name contains the "in" constant
ImporterOptions = __import__("loci.plugins.in.ImporterOptions", globals(), locals(), ['object'], -1)

from ij.io import OpenDialog, DirectoryChooser
from ij import IJ
from ij.plugin import ZProjector, RGBStackMerge
from ij.process import StackConverter

from net.imglib2.img import ImagePlusAdapter
from net.imglib2.img.display.imagej import ImageJFunctions 
from net.imglib2.view import Views

import itertools



import os

def importMSR(path, idxes=None):
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

	if not idxes:
		return (imps)
	else:
		return [imps[i] for i in idxes]

def get_neighborhood(centers, step):
	res = list()
	for x in range(centers[0]-int(step/2), step-int(step/2)+centers[0]):
		for y in range(centers[1]-int(step/2), step-int(step/2)+centers[1]):
			for z in range(centers[2]-int(step/2), step-int(step/2)+centers[2]):
				res.append([x, y, z])
	return res


def wrapFloat(imp):
	StackConverter(imp).convertToGray32()
	return ImageJFunctions.wrapFloat(imp)	

def main2():
	imps = importMSR('/Users/david/Desktop/AutomatedAcquisitions/GM_81C_150s/overviews/3ce7893ba6275ab5988c1395aec5251e_field1.msr') 
	fv1, fv2 = getFeatureVectors(wrapFloat(imps[0]), wrapFloat(imps[1]), [3,72,5])
	print(fv2)

def getFeatureVectors(img1, img2, coords, step=5):
	nbh = get_neighborhood(coords, step)
	#print(nbh)
	ra = Views.extendZero(img1).randomAccess()
	fv1 = list()
	for c in nbh:
		ra.setPosition(c)
		fv1.append(ra.get().getRealFloat())

	ra = Views.extendZero(img2).randomAccess()
	fv2 = list()
	for c in nbh:
		ra.setPosition(c)
		fv2.append(ra.get().getRealFloat())

	return (fv1, fv2)
		

def main():

	dc = DirectoryChooser('pick dir!')

	if not dc.getDirectory():
		return

	path = dc.getDirectory()
	
	csvs = [f for f in os.walk(path).next()[2] if (f.endswith('.csv') and not f.endswith('_features.csv'))]
	print(csvs)

	for csv in csvs:
		pathi = os.path.join(path, csv)
		infd = open(pathi, 'r')
		outfd = open(pathi + '_features.csv', 'w')
		for l in infd.readlines():
			coords = map(int, l.split(',')[-3:])
			filename = os.path.join(path, 'overviews', l.split(',')[0] + '.msr')
			imps = importMSR(filename)
			fv1, fv2 = getFeatureVectors(wrapFloat(imps[0]), wrapFloat(imps[1]), coords)
			outfd.write(','.join(map(str, fv1)) + ',' + ','.join(map(str, fv2)) + '\n')

if __name__ == '__main__':
	main()
