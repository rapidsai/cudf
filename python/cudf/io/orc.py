# Copyright (c) 2019, NVIDIA CORPORATION.

from cudf.dataframe.dataframe import DataFrame
from cudf.utils import ioutils

import pyarrow.orc as orc
import warnings


@ioutils.doc_read_orc()
def read_orc(path, columns=None, **kwargs):
    """{docstring}"""
    warnings.warn("Using CPU via PyArrow to read ORC dataset, this will "
                  "be GPU accelerated in the future")
    orc_file = orc.ORCFile(path)
    pa_table = orc_file.read(columns=columns)
    return DataFrame.from_arrow(pa_table)
