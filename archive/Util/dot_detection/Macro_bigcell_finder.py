from fiji.plugin.trackmate import Model
from fiji.plugin.trackmate import Settings
from fiji.plugin.trackmate import TrackMate
from fiji.plugin.trackmate import SelectionModel
from fiji.plugin.trackmate import Logger
from fiji.plugin.trackmate.detection import LogDetectorFactory
from fiji.plugin.trackmate.tracking.sparselap import SparseLAPTrackerFactory
from fiji.plugin.trackmate.tracking import LAPUtils
from ij import IJ
import fiji.plugin.trackmate.visualization.hyperstack.HyperStackDisplayer as HyperStackDisplayer
import fiji.plugin.trackmate.features.FeatureFilter as FeatureFilter
import sys
import fiji.plugin.trackmate.features.track.TrackDurationAnalyzer as TrackDurationAnalyzer
from loci.common import Region
from loci.plugins.in import ImporterOptions
from loci.plugins import BF
from ij import IJ, ImagePlus, ImageStack
import fiji.plugin.trackmate.Settings as Settings
import fiji.plugin.trackmate.Model as Model
import fiji.plugin.trackmate.SelectionModel as SelectionModel
import fiji.plugin.trackmate.TrackMate as TrackMate
import fiji.plugin.trackmate.Logger as Logger
import fiji.plugin.trackmate.detection.DetectorKeys as DetectorKeys
import fiji.plugin.trackmate.detection.DogDetectorFactory as DogDetectorFactory
import fiji.plugin.trackmate.tracking.sparselap.SparseLAPTrackerFactory as SparseLAPTrackerFactory
import fiji.plugin.trackmate.tracking.LAPUtils as LAPUtils
import fiji.plugin.trackmate.visualization.hyperstack.HyperStackDisplayer as HyperStackDisplayer
from ij.measure import Calibration
import ij.process.AutoThresholder
import sys


def load_msr_w_ser(path, series):
    file = str(path)
    options = ImporterOptions()
    #options.viewHyperstack
    options.setId(file)
    # setSeriesOn(int s, boolean value)
    options.setSeriesOn(series, 1)
    imps = BF.openImagePlus(options)
    for imp in imps:
        return imp

#thresholding  no gui
# TODO: implement me


def just_TrackMate_things(imp, threshold):
    model = Model()
    model.setLogger(Logger.IJ_LOGGER)
    settings = Settings()
    settings.setFrom(imp)
    # Configure detector
    settings.detectorFactory = DogDetectorFactory()
    settings.detectorSettings = {
        DetectorKeys.KEY_DO_SUBPIXEL_LOCALIZATION : True,
        DetectorKeys.KEY_RADIUS : 75.0,
        DetectorKeys.KEY_TARGET_CHANNEL : 1,
        DetectorKeys.KEY_THRESHOLD : float(threshold),
        DetectorKeys.KEY_DO_MEDIAN_FILTERING : False,
    }
    # Configure tracker
    settings.trackerFactory = SparseLAPTrackerFactory()
    settings.trackerSettings = LAPUtils.getDefaultLAPSettingsMap()
    """reduntand
    settings.trackerSettings['LINKING_MAX_DISTANCE'] = 0.0 # muss double sein
    settings.trackerSettings['GAP_CLOSING_MAX_DISTANCE']=0.0 # muss double sein
    settings.trackerSettings['MAX_FRAME_GAP']= 1  #Muss integer sein
    """
    # Add the analyzers for some spot features.
    # You need to configure TrackMate with analyzers that will generate
    # the data you need.
    # Here we just add two analyzers for spot, one that computes generic
    # pixel intensity statistics (mean, max, etc...) and one that computes
    # an estimate of each spot's SNR.
    # The trick here is that the second one requires the first one to be in
    # place. Be aware of this kind of gotchas, and read the docs.
    """ redundant:
    settings.addSpotAnalyzerFactory(SpotIntensityAnalyzerFactory())
    settings.addSpotAnalyzerFactory(SpotContrastAndSNRAnalyzerFactory()
    # Add an analyzer for some track features, such as the track mean speed.
    settings.addTrackAnalyzer(TrackSpeedStatisticsAnalyzer())
    settings.initialSpotFilterValue = 1
    print(str(settings))
    """
    #----------------------
    # Instantiate trackmate
    #----------------------
    trackmate = TrackMate(model, settings)
    #------------
    # Execute all
    #------------
    """redundant
    ok = trackmate.checkInput()
    if not ok:
        sys.exit(str(trackmate.getErrorMessage()))
    """
    ok = trackmate.process()
    if not ok:
        sys.exit(str(trackmate.getErrorMessage()))
    #----------------
    # Display results
    #----------------
    """redundant
    model.getLogger().log('Found ' + str(model.getTrackModel().nTracks(True)) + ' tracks.')
    """
    selectionModel = SelectionModel(model)
    displayer =  HyperStackDisplayer(model, selectionModel, imp)
    displayer.render()
    displayer.refresh()
    return model


def give_back_coords(model):
    coordinatesffs = []
    for spot in model.getSpots().iterable(False):
        coordinatesffs.append((str(spot.getFloatPosition(0)))+ " " +
                              (str(spot.getFloatPosition(1))))
    return coordinatesffs


def save_coords_to_temp(coords, tfile="coords-temp"):
    file = open(str(tfile), 'w')
    file.write(str(coords))
    file.close()


def main():
    series = int(sys.argv[1])
    #size = sys.argv[2]
    threshold = float(sys.argv[3])
    # actual work
    # argparser wont work due to python 2.5 in Jython
    image = load_msr_w_ser("/home/pascal/uni/Bachelorarbeit/DATA/sample2/20160512_k562_001.msr", 0)
    image.show()
    imp = IJ.getImage()
    IJ.run("Auto Threshold", "method=Huang white")
    IJ.run("Gaussian Blur...", "sigma=5")
    IJ.run("Auto Threshold", "method=Otsu white")
    imp.setCalibration(Calibration())
    model = just_TrackMate_things(imp, threshold)
    coordinates = give_back_coords(model)
    save_coords_to_temp(coordinates)
    sys.exit()

main()