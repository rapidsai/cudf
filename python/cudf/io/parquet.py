# Copyright (c) 2019, NVIDIA CORPORATION.

from cudf.dataframe.dataframe import DataFrame
from cudf.utils import ioutils

import pyarrow.parquet as pq
import warnings


@ioutils.doc_read_parquet()
def read_parquet(path, *args, **kwargs):
    """{docstring}"""

    warnings.warn("Using CPU via PyArrow to read Parquet dataset, this will "
                  "be GPU accelerated in the future")
    pa_table = pq.read_pandas(path, *args, **kwargs)
    return DataFrame.from_arrow(pa_table)


@ioutils.doc_to_parquet()
def to_parquet(df, path, *args, **kwargs):
    """{docstring}"""
    warnings.warn("Using CPU via PyArrow to write Parquet dataset, this will "
                  "be GPU accelerated in the future")
    pa_table = df.to_arrow()
    pq.write_to_dataset(pa_table, path, *args, **kwargs)
