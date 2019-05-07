# Copyright (c) 2019, NVIDIA CORPORATION.

from cudf.bindings.parquet import cpp_read_parquet
from cudf.dataframe.dataframe import DataFrame
from cudf.utils import ioutils

import pyarrow.parquet as pq

import warnings


@ioutils.doc_read_parquet_metadata()
def read_parquet_metadata(path):
    """{docstring}"""

    pq_file = pq.ParquetFile(path)

    num_rows = pq_file.metadata.num_rows
    num_row_groups = pq_file.num_row_groups
    col_names = pq_file.schema.names

    return num_rows, num_row_groups, col_names


@ioutils.doc_read_parquet()
def read_parquet(path, engine='cudf', columns=None, row_group=None,
                 skip_rows=None, num_rows=None, strings_to_categorical=False,
                 *args, **kwargs):
    """{docstring}"""

    if engine == 'cudf':
        df = cpp_read_parquet(
            path,
            columns,
            row_group,
            skip_rows,
            num_rows,
            strings_to_categorical
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
