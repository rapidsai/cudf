# Copyright (c) 2020-2022, NVIDIA CORPORATION.

import pandas as pd
from packaging import version

PANDAS_VERSION = version.parse(pd.__version__)
PANDAS_GE_110 = PANDAS_VERSION >= version.parse("1.1")
PANDAS_GE_120 = PANDAS_VERSION >= version.parse("1.2")
PANDAS_LE_122 = PANDAS_VERSION <= version.parse("1.2.2")
PANDAS_GE_130 = PANDAS_VERSION >= version.parse("1.3.0")
PANDAS_GE_133 = PANDAS_VERSION >= version.parse("1.3.3")
PANDAS_GE_134 = PANDAS_VERSION >= version.parse("1.3.4")
PANDAS_LT_140 = PANDAS_VERSION < version.parse("1.4.0")
PANDAS_GE_150 = PANDAS_VERSION >= version.parse("1.5.0")
