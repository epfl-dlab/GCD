from src.prompting.legacy.EL.demo import DEMOs as DEMOs_EL
from src.prompting.legacy.EL.demo_long import DEMOs as DEMOs_EL_long
from src.prompting.legacy.EL.demo_canonical import DEMOs as DEMOs_EL_canonical
from src.prompting.legacy.IE.demo_sc import DEMOs as DEMOs_IE_sc
from src.prompting.legacy.IE.demo_fe import DEMOs as DEMOs_IE_fe
from src.prompting.legacy.CP.demo_ptb import DEMOs as DEMOs_CP_ptb
from src.prompting.legacy.CP.demo_ptb64 import DEMOs as DEMOs_CP_ptb64
from src.prompting.legacy.CP.demo_ptbM import DEMOs as DEMOs_CP_ptbM

DEMOs = {
    "el": {
        "short": DEMOs_EL,
        "long": DEMOs_EL_long,
        "canonical": DEMOs_EL_canonical,
    },
    "ie": {"sc": DEMOs_IE_sc, "fe": DEMOs_IE_fe},
    "cp": {
        "ptb": DEMOs_CP_ptb,
        "ptbM": DEMOs_CP_ptbM,
        "ptb64": DEMOs_CP_ptb64,
    },
}
