from fiji.plugin.trackmate import Model
from fiji.plugin.trackmate import Settings
from fiji.plugin.trackmate import TrackMate
from fiji.plugin.trackmate import SelectionModel
from fiji.plugin.trackmate import Logger
from ij import IJ
from loci.plugins.in import ImporterOptions
from loci.plugins import BF
from ij import ImagePlus, ImageStack
import fiji.plugin.trackmate.Model as Model
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
    """
    Loads a .msr file from a given path. 1 Series of a .msr file is loaded
    :param path: path to the image
    :param series: series of the .msr file
    :return: returns image es "imp" variable
    """
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
    """
    Tracks "interesting" spots/ returns their coordinates in X,Y,Z
    :param imp: Image as imp variable
    :param threshold: value for thresholding filter
    :return: returns a "model" object with the needed information
    """
    model = Model()
    model.setLogger(Logger.IJ_LOGGER)
    settings = Settings()
    settings.setFrom(imp)
    # Configure detector
    settings.detectorFactory = DogDetectorFactory()
    settings.detectorSettings = {
        DetectorKeys.KEY_DO_SUBPIXEL_LOCALIZATION : True,
        DetectorKeys.KEY_RADIUS : 15.0,
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
    """
    Uses the "model" object and extracts the coordinates of the "interesting" spots
    :param model: model object produced by TrackMate
    :return: returns coordinates as float (in pixel)
    """
    coordinatesffs = []
    for spot in model.getSpots().iterable(False):
        coordinatesffs.append((str(spot.getFloatPosition(0)))+ " " +
                              (str(spot.getFloatPosition(1))))
    return coordinatesffs


def save_coords_to_temp(coords, tfile="coords-temp"):
    """
    Saves the "coords" variable to ./coords-temp file
    :param coords: coordinates saved in a list
    :param tfile: temporary file for saving the coordinates
    :return: None
    """
    file = open(str(tfile), 'w')
    file.write(str(coords))
    file.close()


def main():

    #x#x#x#x#x#x#x#x#x#x#x#x#x#x#x#x
    # parsing arguments
    path_to_image = str(sys.argv[1])
    series = int(sys.argv[2])
    threshold = float(sys.argv[3])
    #size = sys.argv[4]
    #x#x#x#x#x#x#x#x#x#x#x#x#x#x#x#x


    #x#x#x#x#x#x#x#x#x#x#x#x#x#x#x#x
    # parsing arguments - alternative for --macro use - ifloop?
    #path_to_image = str(sys.argv[2])
    #series = int(sys.argv[3])
    #threshold = float(sys.argv[4])
    #size = sys.argv[5]
    #x#x#x#x#x#x#x#x#x#x#x#x#x#x#x#x

    # actual work
    image = load_msr_w_ser(str(path_to_image), series)
    image.show()
    imp = IJ.getImage()
    IJ.run("Auto Threshold", "method=MaxEntropy white")
    imp.setCalibration(Calibration())
    model = just_TrackMate_things(imp, threshold)
    coordinates = give_back_coords(model)
    save_coords_to_temp(coordinates)
    sys.exit()


main()
