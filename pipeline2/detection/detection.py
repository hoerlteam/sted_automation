from spot_util import pair_finder_inner, detect_blobs
import numpy as np
import collections

from ..util import filter_dict

from display_util import draw_detections_2c, draw_detections_1c

class SimpleSingleChannelSpotDetector():
    '''
    simple LoG-based spot detector in one channel
    '''
    def __init__(self, dataSource, sigmas, threshold, channel=0, medianThreshold=3, medianRadius=5):
        self.dataSource = dataSource
        self.sigmas = sigmas
        self.threshold = threshold
        self.channel = channel
        self.medianThreshold = medianThreshold
        self.medianRadius = medianRadius
        self.plotDetections = False
        self.verbose = False
        
        
    def withPlotDetections(self, plotDetections=True):
        self.plotDetections = plotDetections
        return self

    
    def withVerbose(self, verbose=True):
        self.verbose = verbose
        return self
    
    
    def doPlot(self, spots_pixel, img):
        draw_detections_1c(img, [s[-1::-1] for s in spots_pixel], None, 0, 3)

    def correctForOffset(self, locs, setts):
        offsOld = np.array([filter_dict(
            setts, 'ExpControl/scan/range/{}/off'.format(c), False) for c in ['x', 'y', 'z']], dtype=float)

        lensOld = np.array([filter_dict(
            setts, 'ExpControl/scan/range/{}/len'.format(c), False) for c in ['x', 'y', 'z']], dtype=float)

        pszOld = np.array([filter_dict(
            setts, 'ExpControl/scan/range/{}/psz'.format(c), False) for c in ['x', 'y', 'z']], dtype=float)

        res = []
        for loc in locs:
            locT = np.array(loc, dtype=float)
            res.append(_correct_offset(loc, offsOld, lensOld, pszOld))
        return res
    
    
    def get_locations(self):
        data = self.dataSource.get_data()
        if (data.numConfigurations < 1) or (data.numImages(0) < 1):
            raise ValueError(
                'no images present. TODO: fail gracefully/skip here')
        
        img = data.data[0][self.channel][0, :, :, :]
        

        # make float
        img = np.array(img, np.float)
        setts = data.measurementSettings[0]

        if not isinstance(self.sigmas, (collections.Sequence, np.ndarray)):
            self.sigmas = [self.sigmas, self.sigmas, self.sigmas]
            
        # NB: will give locs in z,y,x -> invert
        locs = detect_blobs(img, self.sigmas, self.threshold, False, self.medianThreshold,
                                     self.medianRadius)
        locs = [l[-1::-1] for l in locs]

        if self.verbose:
            print(self.__class__.__name__ + ': found {} spots. pixel coordinates:'.format(len(locs)))
            for loc in locs:
                print(loc)

        # plot
        if self.plotDetections:
            self.doPlot(locs, img)

        corrected = self.correctForOffset(locs, setts)

        if self.verbose:
            print(self.__class__.__name__ + ': found {} spots. offsets:'.format(len(locs)))
            for locC in corrected:
                print(locC)

        return corrected
    

class LegacySpotPairFinder():
    """
    wrapper for the 'old' spot pair detector
    get_locations will return a list of coordinate lists
    of scan coordinates (stage coordinates are ignored)
    """

    def __init__(self, dataSource, sigma, thresholds, medianThresholds=[3, 3], medianRadius=5):
        self.dataSource = dataSource
        self.sigma = sigma
        self.thresholds = thresholds
        self.medianThresholds = medianThresholds
        self.medianRadius = medianRadius
        self.plotDetections = False
        self.verbose = False

    def withPlotDetections(self, plotDetections=True):
        self.plotDetections = plotDetections
        return self

    def withVerbose(self, verbose=True):
        self.verbose = verbose
        return self

    def doPlot(self, pairsPixel, stack1, stack2):
        draw_detections_2c(stack1, stack2, [s[-1::-1] for s in pairsPixel], [1, 10], 0, 3)

    def correctForOffset(self, pairsPixel, setts):
        offsOld = np.array([filter_dict(
            setts, 'ExpControl/scan/range/{}/off'.format(c), False) for c in ['x', 'y', 'z']], dtype=float)

        lensOld = np.array([filter_dict(
            setts, 'ExpControl/scan/range/{}/len'.format(c), False) for c in ['x', 'y', 'z']], dtype=float)

        pszOld = np.array([filter_dict(
            setts, 'ExpControl/scan/range/{}/psz'.format(c), False) for c in ['x', 'y', 'z']], dtype=float)

        res = []
        for pair in pairsPixel:
            pairT = np.array(pair, dtype=float)
            res.append(_correct_offset(pairT, offsOld, lensOld, pszOld))
        return res

    def get_locations(self):
        data = self.dataSource.get_data()
        if (data.numConfigurations < 1) or (data.numImages(0) < 2):
            raise ValueError(
                'too few images for LegacySpotPairFinder. The RichData provided needs to have two images in the first configuration.')
        stack1 = data.data[0][0][0, :, :, :]
        stack2 = data.data[0][1][0, :, :, :]

        # make float
        stack1 = np.array(stack1, np.float)
        stack2 = np.array(stack2, np.float)

        setts = data.measurementSettings[0]

        pairsRaw = pair_finder_inner(stack1, stack2, self.sigma, self.thresholds, True, False, self.medianThresholds,
                                     self.medianRadius)

        if self.verbose:
            print(self.__class__.__name__ + ': found {} spot pairs. pixel coordinates:'.format(len(pairsRaw)))
            for pr in pairsRaw:
                print(pr)

        # plot
        if self.plotDetections:
            self.doPlot(pairsRaw, stack1, stack2)

        corrected = self.correctForOffset(pairsRaw, setts)

        if self.verbose:
            print(self.__class__.__name__ + ': found {} spot pairs. offsets:'.format(len(pairsRaw)))
            for pc in corrected:
                print(pc)

        return corrected


class ZDCSpotPairFinder(LegacySpotPairFinder):


    def correctForOffset(self, pairsPixel, setts):
        offsOld = np.array([filter_dict(
            setts, 'ExpControl/scan/range/{}/off'.format(c), False) for c in ['x', 'y', 'z']], dtype=float)

        # we use the coarse offset here
        offsOld[2] = filter_dict(setts, 'ExpControl/scan/range/offsets/coarse/z/g_off', False)

        print(offsOld)
        lensOld = np.array([filter_dict(
            setts, 'ExpControl/scan/range/{}/len'.format(c), False) for c in ['x', 'y', 'z']], dtype=float)

        print(lensOld)
        pszOld = np.array([filter_dict(
            setts, 'ExpControl/scan/range/{}/psz'.format(c), False) for c in ['x', 'y', 'z']], dtype=float)

        print(pszOld)
        res = []
        for pair in pairsPixel:
            pairT = np.array(pair, dtype=float)
            #res.append(list(offsOld - (lensOld / 2) + pairT * pszOld))
            res.append(_correct_offset(pairT, offsOld, lensOld, pszOld))
        return res


def _correct_offset(x, off, length, psz):
    """
    correct pixel coordinates x to world coordinates
    :param x: pixel coordinates (array-like)
    :param off: Imspector metadata offset (array-like)
    :param length: Imspector metadata FOV-length (array-like)
    :param psz: Imspector metadata pixel-size (array-like)
    :return: x in world coordinates (array-like)
    """
    return (np.array(off, dtype=float) - (np.array(length, dtype=float)
            - np.array(psz, dtype=float)) / 2.0 + np.array(x, dtype=float) * np.array(psz, dtype=float))