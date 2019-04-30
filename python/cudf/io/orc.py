# Copyright (c) 2019, NVIDIA CORPORATION.

from cudf.bindings.orc import cpp_read_orc
from cudf.dataframe.dataframe import DataFrame
from cudf.utils import ioutils

import pyarrow.orc as orc

import warnings


@ioutils.doc_read_orc()
def read_orc(path, engine='cudf', columns=None, skip_rows=None,
             num_rows=None):
    """{docstring}"""

    if engine == 'cudf':
        df = cpp_read_orc(
            path,
            columns,
            skip_rows,
            num_rows
        )
    else:
        warnings.warn("Using CPU via PyArrow to read ORC dataset.")
        orc_file = orc.ORCFile(path)
        pa_table = orc_file.read(columns=columns)
        df = DataFrame.from_arrow(pa_table)

    return df
