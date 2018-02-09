import specpy
import numpy as np
import json
from matplotlib import pyplot
import time

from skimage.feature import blob_dog

from unittest.mock import MagicMock

from pipeline2 import AcquisitionPipeline, PipelineLevels, DefaultNameHandler
from pipeline2.util import gen_json, dump_JSON
from pipeline2.imspector.imspector import get_current_stage_coords
from pipeline2.taskgeneration import (SpiralOffsetGenerator, JSONFileConfigLoader,
                                      DefaultStageOffsetsSettingsGenerator,
                                      AcquisitionTaskGenerator,
                                      DefaultLocationRemover, DefaultLocationKeeper,
                                      NewestDataSelector,
                                      NewestSettingsSelector,
                                      ZDCOffsetSettingsGenerator)
from pipeline2.imspector import ImspectorConnection
from pipeline2.stoppingcriteria import TimedStoppingCriterion
from pipeline2.detection import ZDCSpotPairFinder

def main():
    levels = PipelineLevels('overview', 'detail')

    im = specpy.Imspector()

    c = get_current_stage_coords(im)
    sp = SpiralOffsetGenerator().withVerbose()
    sp.withStart(c)
    sp.withZOffset(c[2] + 5e-6)

    atg = (AcquisitionTaskGenerator(levels.overview,
                                    DefaultLocationRemover(JSONFileConfigLoader(
                                        ['C:/Users/RESOLFT/Desktop/config_json/zdc_overview.json'])),
                                    DefaultStageOffsetsSettingsGenerator(sp))
           .withDelay(2.0))

    pipeline = AcquisitionPipeline('1')

    detector = ZDCSpotPairFinder(NewestDataSelector(pipeline, levels.overview), sigma=3, thresholds=[0.01, 0.01])
    detector.withVerbose()
    detector.withPlotDetections()

    atg_detail = (AcquisitionTaskGenerator(levels.detail,
                                           DefaultLocationRemover(JSONFileConfigLoader(
                                               ['C:/Users/RESOLFT/Desktop/config_json/zdc_detail.json'])),
                                           DefaultLocationKeeper(NewestSettingsSelector(pipeline, levels.overview)),
                                           ZDCOffsetSettingsGenerator(detector)
                                           )
            .withDelay(2.0))


    pipeline.withPipelineLevels(levels)
    pipeline.withNameHandler(DefaultNameHandler('C:/Users//RESOLFT/Desktop/TEST_GEN/', levels))
    pipeline.withImspectorConnection(ImspectorConnection(im))

    pipeline.withCallbackAtLevel(atg, levels.overview)
    pipeline.withCallbackAtLevel(atg_detail, levels.overview)
    pipeline.withAddedStoppingCondition(TimedStoppingCriterion(300))

    atg(pipeline)


    pipeline.run()

if __name__ == '__main__':
    main()