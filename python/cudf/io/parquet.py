# Copyright (c) 2019, NVIDIA CORPORATION.

from cudf.bindings.parquet import cpp_read_parquet
from cudf.dataframe.dataframe import DataFrame
from cudf.utils import ioutils

import pyarrow.parquet as pq

import warnings


@ioutils.doc_read_parquet()
def read_parquet(path, engine='cudf', columns=None, *args, **kwargs):
    """{docstring}"""

    if engine == 'cudf':
        df = cpp_read_parquet(
            path,
            columns
        )
    else:
        warnings.warn("Using CPU via PyArrow to read Parquet dataset.")
        pa_table = pq.read_pandas(
            path,
            columns=columns,
            *args,
            **kwargs
        )
        df = DataFrame.from_arrow(pa_table)

    return df


@ioutils.doc_to_parquet()
def to_parquet(df, path, *args, **kwargs):
    """{docstring}"""
    warnings.warn("Using CPU via PyArrow to write Parquet dataset, this will "
                  "be GPU accelerated in the future")
    pa_table = df.to_arrow()
    pq.write_to_dataset(pa_table, path, *args, **kwargs)
