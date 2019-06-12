# Copyright (c) 2019, NVIDIA CORPORATION.

from cudf.bindings.orc import cpp_read_orc
from cudf.dataframe.dataframe import DataFrame
from cudf.utils import ioutils

import pyarrow.orc as orc

import warnings


@ioutils.doc_read_orc_metadata()
def read_orc_metadata(path):
    """{docstring}"""

    orc_file = orc.ORCFile(path)

    num_rows = orc_file.nrows
    num_stripes = orc_file.nstripes
    col_names = orc_file.schema.names

    return num_rows, num_stripes, col_names


@ioutils.doc_read_orc()
def read_orc(path, engine='cudf', columns=None, stripe=None, skip_rows=None,
             num_rows=None, use_index=True):
    """{docstring}"""

    if engine == 'cudf':
        df = cpp_read_orc(
            path,
            columns,
            stripe,
            skip_rows,
            num_rows,
            use_index
        )
    else:
        warnings.warn("Using CPU via PyArrow to read ORC dataset.")
        orc_file = orc.ORCFile(path)
        pa_table = orc_file.read(columns=columns)
        df = DataFrame.from_arrow(pa_table)

    return df
