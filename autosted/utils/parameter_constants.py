OFFSET_STAGE_PARAMETERS = (
    "ExpControl/scan/range/coarse_z/off",
    "ExpControl/scan/range/coarse_y/off",
    "ExpControl/scan/range/coarse_x/off",
)

OFFSET_STAGE_GLOBAL_PARAMETERS = (
    "ExpControl/scan/range/coarse_z/g_off",
    "ExpControl/scan/range/coarse_y/g_off",
    "ExpControl/scan/range/coarse_x/g_off",
)

OFFSET_SCAN_PARAMETERS = (
    "ExpControl/scan/range/z/off",
    "ExpControl/scan/range/y/off",
    "ExpControl/scan/range/x/off",
)

OFFSET_SCAN_GLOBAL_PARAMETERS = (
    "ExpControl/scan/range/z/g_off",
    "ExpControl/scan/range/y/g_off",
    "ExpControl/scan/range/x/g_off",
)

LOCATION_PARAMETERS = (
    OFFSET_STAGE_PARAMETERS
    + OFFSET_STAGE_GLOBAL_PARAMETERS
    + OFFSET_SCAN_PARAMETERS
    + OFFSET_SCAN_GLOBAL_PARAMETERS
    + ("OlympusIX/stage", "OlympusIX/scanrange")
)

FOV_LENGTH_PARAMETERS = (
    "ExpControl/scan/range/z/len",
    "ExpControl/scan/range/y/len",
    "ExpControl/scan/range/x/len",
)

PIXEL_SIZE_PARAMETERS = (
    "ExpControl/scan/range/z/psz",
    "ExpControl/scan/range/y/psz",
    "ExpControl/scan/range/x/psz",
)

# directions of stage and scan coordinates relative to image pixels
# 1: larger pixel coord <-> larger world coord
# -1: larger pixel coord <-> smaller world coord
# NOTE: axes may be flipped in Imspector, Lightbox tiling seems to require flipped x-stage coords
DIRECTION_STAGE = (1, 1, -1)
DIRECTION_SCAN = (1, 1, 1)
