# Copyright (c) 2018, NVIDIA CORPORATION.
from cudf import dataframe

<<<<<<< HEAD
from cudf.dataframe import DataFrame
from cudf.dataframe import Index
from cudf.dataframe import Series
from cudf.multi import concat
from cudf.io import read_csv
from cudf.settings import set_options
=======
from cudf.dataframe import DataFrame, from_pandas   # noqa: F401
from cudf.dataframe import Index       # noqa: F401
from cudf.dataframe import Series      # noqa: F401
from cudf.multi import concat          # noqa: F401
from cudf.io import read_csv           # noqa: F401
from cudf.settings import set_options  # noqa: F401
>>>>>>> 1a6606daadf547636e94ae82df78fa4be47c21bf


# Versioneer
from ._version import get_versions
__version__ = get_versions()['version']
del get_versions
