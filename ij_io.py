import re, os, sys, itertools

from ij import IJ, ImagePlus
from loci.plugins import BF
from ij.plugin import ZProjector, RGBStackMerge

from java.util.concurrent import Callable, Executors

import sys

# manually import ImporterOptions, as the package name contains the "in" constant
ImporterOptions = __import__("loci.plugins.in.ImporterOptions", globals(), locals(), ['object'], -1)


class ResaveThread(Callable):
    def __init__(self, in_path, out_path, series, func):
        self.in_path = in_path
        self.out_path = out_path
        self.series = series
        self.func = func

    def call(self):
        self.func(self.in_path, self.out_path, self.series)
        print('finished file ' + self.in_path)

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


def resave_msr_as_jpeg_sum_projection(path, outpath, series):
    imps = importMSR(path)


    if series != None:
        imps = [imps[i] for i in series]

    impsProjected = []
    for impi in imps:
        zp = ZProjector(impi)
        zp.setMethod(ZProjector.SUM_METHOD)
        zp.doProjection()
        impsProjected.append(zp.getProjection())

    for impi in impsProjected:
        impi.resetDisplayRange()
        impi.updateImage()

    impComp = RGBStackMerge.mergeChannels(impsProjected, False)
    IJ.saveAs(impComp, 'JPEG', outpath)


def resave_msr_folder_as_jpeg_sum_projection(path, allseries=False, suffix='.msr'):

    files = [os.path.join(path, f) for f in os.walk(path).next()[2] if f.endswith(suffix)]

    MAX_CONCURRENT = 8
    service = Executors.newFixedThreadPool(MAX_CONCURRENT)
    savers = list()

    for i in range(len(files)):
        savers.append(ResaveThread(files[i], files[i] + '.jpg',  None if allseries else range(2) if not 'sted' in files[i].split(os.sep)[-1] else range(2,4), resave_msr_as_jpeg_sum_projection))

    futures = service.invokeAll(savers)

    for f in futures:
        f.get()

    print('-- ALL DONE --')
    service.shutdown()

def resave_msr_as_tiff(in_path, out_path=None, series=None):

    imps = importMSR(in_path)

    if series != None:
        imps = [imps[i] for i in series]

    for i in range(len(imps)):
        IJ.saveAsTiff(imps[i], out_path + 'ch' + str(i) + '.tif')

def resave_msr_folder_as_tiff(path, allseries=False):

    # create output directory
    if not os.path.exists(os.path.join(path, "tiffs")):
        os.makedirs(os.path.join(path, "tiffs"));

    files = [f for f in os.walk(path).next()[2] if f.endswith('.msr')]

    MAX_CONCURRENT = 8
    service = Executors.newFixedThreadPool(MAX_CONCURRENT)
    savers = list()

    for fi in range(len(files)):
        f = os.path.join(path, files[fi])
        outpreffix = os.path.join(path, 'tiffs', files[fi])

        savers.append(ResaveThread(f, outpreffix, None if allseries else range(2) if not ('sted' in files[fi]) else range(2,4), resave_msr_as_tiff))

    futures = service.invokeAll(savers)

    for f in futures:
        f.get()


    print('-- ALL DONE --')
    service.shutdown()