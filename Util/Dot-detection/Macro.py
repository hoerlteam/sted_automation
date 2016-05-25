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
import fiji.plugin.trackmate.features.FeatureFilter as FeatureFilter
import fiji.plugin.trackmate.features.FeatureAnalyzer as FeatureAnalyzer
import fiji.plugin.trackmate.features.spot.SpotContrastAndSNRAnalyzerFactory as SpotContrastAndSNRAnalyzerFactory
import fiji.plugin.trackmate.action.ExportStatsToIJAction as ExportStatsToIJAction
import fiji.plugin.trackmate.io.TmXmlReader as TmXmlReader
import fiji.plugin.trackmate.action.ExportTracksToXML as ExportTracksToXML
import fiji.plugin.trackmate.io.TmXmlWriter as TmXmlWriter
import fiji.plugin.trackmate.features.ModelFeatureUpdater as ModelFeatureUpdater
import fiji.plugin.trackmate.features.SpotFeatureCalculator as SpotFeatureCalculator
import fiji.plugin.trackmate.features.spot.SpotContrastAndSNRAnalyzer as SpotContrastAndSNRAnalyzer
import fiji.plugin.trackmate.features.spot.SpotIntensityAnalyzerFactory as SpotIntensityAnalyzerFactory
import fiji.plugin.trackmate.features.track.TrackSpeedStatisticsAnalyzer as TrackSpeedStatisticsAnalyzer
import fiji.plugin.trackmate.util.TMUtils as TMUtils

def load_msr_w_ser():
	# Hier kann man noch den path und die series als variablen angeben
	file = "/mnt/storage/Pascal/Desktop/Uni/Bachelorarbeit/DATA/CF610sample/20160513_k562_HS2_CF610_003.msr"
	options = ImporterOptions()
	options.viewHyperstack
	options.setId(file)
	# setSeriesOn(int s, boolean value)
	options.setSeriesOn(1,1)
	imps = BF.openImagePlus(options)
	for imp in imps:
	    return imp
	    #imp.show #--alternative

image = load_msr_w_ser()
image.show()
imp = IJ.getImage()
IJ.run("Auto Threshold", "method=MaxEntropy white");


model = Model()
model.setLogger(Logger.IJ_LOGGER)   
settings = Settings()
settings.setFrom(imp)
      
# Configure detector
settings.detectorFactory = DogDetectorFactory()
settings.detectorSettings = {
    DetectorKeys.KEY_DO_SUBPIXEL_LOCALIZATION : True,
    DetectorKeys.KEY_RADIUS : 0.3,
    DetectorKeys.KEY_TARGET_CHANNEL : 1,
    DetectorKeys.KEY_THRESHOLD : 1.,
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
settings.addSpotAnalyzerFactory(SpotContrastAndSNRAnalyzerFactory())
   
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

#print(model.getSpots())
for spot in model.getSpots().iterable(False):
	print(str(spot.getFloatPosition(0))+ str(spot.getFloatPosition(1)))
# The feature model, that stores edge and track features.
