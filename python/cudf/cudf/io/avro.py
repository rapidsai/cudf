# Copyright (c) 2019, NVIDIA CORPORATION.

import warnings

import fastavro as fa
import pandas as pd

from cudf.bindings.avro import cpp_read_avro
from cudf.dataframe.dataframe import DataFrame
from cudf.utils import ioutils


@ioutils.doc_read_avro()
def read_avro(
    filepath_or_buffer,
    engine="cudf",
    columns=None,
    skip_rows=None,
    num_rows=None,
):
    """{docstring}"""

    filepath_or_buffer, compression = ioutils.get_filepath_or_buffer(
        filepath_or_buffer, None
    )
    if compression is not None:
        ValueError("URL content-encoding decompression is not supported")

    if engine == "cudf":
        df = cpp_read_avro(filepath_or_buffer, columns, skip_rows, num_rows)
    else:
        warnings.warn("Using CPU via fastavro to read Avro dataset.")
        if ioutils.is_file_like(filepath_or_buffer):
            reader = fa.reader(filepath_or_buffer)
        else:
            reader = fa.reader(open(filepath_or_buffer, "rb"))

        pdf = pd.DataFrame.from_records(reader, columns=columns)
        df = DataFrame.from_pandas(pdf)

    return df
