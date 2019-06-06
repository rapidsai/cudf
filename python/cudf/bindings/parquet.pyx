# Copyright (c) 2019, NVIDIA CORPORATION.

# cython: profile=False
# distutils: language = c++
# cython: embedsignature = True
# cython: language_level = 3

from .cudf_cpp cimport *
from .cudf_cpp import *
from cudf.bindings.parquet cimport *
from libc.stdlib cimport free
from libcpp.vector cimport vector

from cudf.dataframe.column import Column
from cudf.dataframe.numerical import NumericalColumn
from cudf.dataframe.dataframe import DataFrame
from cudf.dataframe.datetime import DatetimeColumn
from cudf.utils import ioutils
from cudf.bindings.nvtx import nvtx_range_push, nvtx_range_pop
from librmm_cffi import librmm as rmm

import nvstrings
import numpy as np
import collections.abc
import errno
import os


cpdef cpp_read_parquet(path, columns=None, row_group=None, skip_rows=None,
                       num_rows=None, strings_to_categorical=False):
    """
    Cython function to call into libcudf API, see `read_parquet`.

    See Also
    --------
    cudf.io.parquet.read_parquet
    cudf.io.parquet.to_parquet
    """

    # Setup arguments
    cdef pq_read_arg pq_reader = pq_read_arg()

    if not os.path.isfile(path) or not os.path.exists(path):
        raise FileNotFoundError(
            errno.ENOENT, os.strerror(errno.ENOENT), path
        )
    path = str(path).encode()
    source_ptr = <char*>path
    pq_reader.source_type = FILE_PATH
    pq_reader.source = source_ptr

    usecols = columns
    arr_cols = []
    pq_reader.use_cols_len = 0
    cdef vector[const char*] vector_use_cols
    if usecols is not None:
        for col in usecols:
            arr_cols.append(str(col).encode())
        pq_reader.use_cols_len = len(usecols)

    vector_use_cols = arr_cols
    pq_reader.use_cols = vector_use_cols.data()

    if row_group is not None:
        pq_reader.row_group = row_group
    else:
        pq_reader.row_group = -1

    if skip_rows is not None:
        pq_reader.skip_rows = skip_rows
    else:
        pq_reader.skip_rows = 0

    if num_rows is not None:
        pq_reader.num_rows = num_rows
    else:
        pq_reader.num_rows = -1

    pq_reader.strings_to_categorical = strings_to_categorical

    # Call read_parquet
    with nogil:
        result = read_parquet(&pq_reader)

    check_gdf_error(result)

    out = pq_reader.data
    if out is NULL:
        raise ValueError("Failed to parse Parquet")

    # Extract parsed columns
    outcols = []
    new_names = []
    for i in range(pq_reader.num_cols_out):
        data_mem, mask_mem = gdf_column_to_column_mem(out[i])
        outcols.append(Column.from_mem_views(data_mem, mask_mem))
        new_names.append(out[i].col_name.decode())
        free(out[i].col_name)
        free(out[i])

    # Construct dataframe from columns
    df = DataFrame()
    for k, v in zip(new_names, outcols):
        df[k] = v

    # Set column to use as row indexes if available
    if pq_reader.index_col is not NULL:
        df = df.set_index(df.columns[pq_reader.index_col[0]])
        free(pq_reader.index_col)
        new_index_name = pa.pandas_compat._backwards_compatible_index_name(
            df.index.name, df.index.name
        )
        df.index.name = new_index_name

    return df
