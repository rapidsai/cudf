# Copyright (c) 2020-2021, NVIDIA CORPORATION.

import pandas as pd
from packaging import version

PANDAS_VERSION = version.parse(pd.__version__)
PANDAS_GE_100 = PANDAS_VERSION >= version.parse("1.0")
PANDAS_GE_110 = PANDAS_VERSION >= version.parse("1.1")
PANDAS_GE_120 = PANDAS_VERSION >= version.parse("1.2")
PANDAS_LE_122 = PANDAS_VERSION <= version.parse("1.2.2")
