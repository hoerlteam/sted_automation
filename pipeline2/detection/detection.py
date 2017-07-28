from spot_util import pair_finder_inner
import numpy as np
from ..util import filter_dict

from display_util import draw_detections_2c


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

    def withPlotDetections(self, plotDetections=True):
        self.plotDetections = plotDetections
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
            res.append(list(offsOld - (lensOld / .2) + pairT * pszOld))
        return res

    def get_locations(self):
        data = self.dataSource.get_data()
        if (data.numConfigurations < 1) or (data.numImages(0) < 2):
            raise ValueError(
                'too few images for LegacySpotPairFinder. The RichData provided needs to have two images in the first configuration.')
        stack1 = data.data[0][0][0, :, :, :]
        stack2 = data.data[0][1][0, :, :, :]

        setts = data.measurementSettings[0]

        pairsRaw = pair_finder_inner(stack1, stack2, self.sigma, self.thresholds, True, False, self.medianThresholds,
                                     self.medianRadius)
        # plot
        if self.plotDetections:
            self.doPlot(pairsRaw, stack1, stack2)

        return self.correctForOffset(pairsRaw, setts)


class ZDCSpotPairFinder(LegacySpotPairFinder):


    def correctForOffset(self, pairsPixel, setts):
        offsOld = np.array([filter_dict(
            setts, 'ExpControl/scan/range/{}/off'.format(c), False) for c in ['x', 'y', 'z']], dtype=float)

        # we use the coarse offset here
        offsOld[2] = filter_dict(setts, 'ExpControl/scan/range/offsets/coarse/z/g_off', False)

        lensOld = np.array([filter_dict(
            setts, 'ExpControl/scan/range/{}/len'.format(c), False) for c in ['x', 'y', 'z']], dtype=float)

        pszOld = np.array([filter_dict(
            setts, 'ExpControl/scan/range/{}/psz'.format(c), False) for c in ['x', 'y', 'z']], dtype=float)

        res = []
        for pair in pairsPixel:
            pairT = np.array(pair, dtype=float)
            res.append(list(offsOld - (lensOld / .2) + pairT * pszOld))
        return res


