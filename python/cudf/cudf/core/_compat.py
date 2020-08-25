# Copyright (c) 2020, NVIDIA CORPORATION.
import numba
import pandas as pd
from packaging import version

PANDAS_VERSION = version.parse(pd.__version__)
PANDAS_GE_100 = PANDAS_VERSION >= version.parse("1.0")
PANDAS_GE_110 = PANDAS_VERSION >= version.parse("1.1")

NUMBA_VERSION = version.parse(numba.__version__)
NUMBA_LE_0501 = NUMBA_VERSION <= version.parse("0.50.1")
