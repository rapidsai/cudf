# Copyright (c) 2020-2023, NVIDIA CORPORATION.

import pandas as pd
from packaging import version

PANDAS_VERSION = version.parse(pd.__version__)
PANDAS_GE_133 = PANDAS_VERSION >= version.parse("1.3.3")
PANDAS_GE_134 = PANDAS_VERSION >= version.parse("1.3.4")
PANDAS_LT_140 = PANDAS_VERSION < version.parse("1.4.0")
PANDAS_GE_150 = PANDAS_VERSION >= version.parse("1.5.0")
PANDAS_LT_153 = PANDAS_VERSION < version.parse("1.5.3")
PANDAS_EQ_200 = PANDAS_VERSION == version.parse("2.0.0")
PANDAS_GE_200 = PANDAS_VERSION >= version.parse("2.0.0")
PANDAS_GE_210 = PANDAS_VERSION >= version.parse("2.1.0")
PANDAS_GE_220 = PANDAS_VERSION >= version.parse("2.2.0")
