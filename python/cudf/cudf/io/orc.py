# Copyright (c) 2019, NVIDIA CORPORATION.

import warnings

import pyarrow.orc as orc

from cudf.bindings.orc import cpp_read_orc
from cudf.dataframe.dataframe import DataFrame
from cudf.utils import ioutils


@ioutils.doc_read_orc_metadata()
def read_orc_metadata(path):
    """{docstring}"""

    orc_file = orc.ORCFile(path)

    num_rows = orc_file.nrows
    num_stripes = orc_file.nstripes
    col_names = orc_file.schema.names

    return num_rows, num_stripes, col_names


@ioutils.doc_read_orc()
def read_orc(
    filepath_or_buffer,
    engine="cudf",
    columns=None,
    stripe=None,
    skip_rows=None,
    num_rows=None,
    use_index=True,
):
    """{docstring}"""

    filepath_or_buffer, compression = ioutils.get_filepath_or_buffer(
        filepath_or_buffer, None
    )
    if compression is not None:
        ValueError("URL content-encoding decompression is not supported")

    if engine == "cudf":
        df = cpp_read_orc(
            filepath_or_buffer, columns, stripe, skip_rows, num_rows, use_index
        )
    else:
        warnings.warn("Using CPU via PyArrow to read ORC dataset.")
        orc_file = orc.ORCFile(filepath_or_buffer)
        pa_table = orc_file.read(columns=columns)
        df = DataFrame.from_arrow(pa_table)

    return df
