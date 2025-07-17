# Copyright (c) 2020-2025, NVIDIA CORPORATION.

import pandas as pd
from packaging import version

PANDAS_CURRENT_SUPPORTED_VERSION = version.parse("2.3.1")
PANDAS_VERSION = version.parse(pd.__version__)


PANDAS_GE_210 = PANDAS_VERSION >= version.parse("2.1.0")
PANDAS_GT_214 = PANDAS_VERSION > version.parse("2.1.4")
PANDAS_GE_220 = PANDAS_VERSION >= version.parse("2.2.0")
PANDAS_GE_230 = PANDAS_VERSION >= version.parse("2.3.0")
PANDAS_LT_300 = PANDAS_VERSION < version.parse("3.0.0")
