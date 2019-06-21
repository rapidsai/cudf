# Copyright (c) 2019, NVIDIA CORPORATION.

# cython: profile=False
# distutils: language = c++
# cython: embedsignature = True
# cython: language_level = 3

from .cudf_cpp cimport *
from .cudf_cpp import *
from cudf.bindings.parquet cimport reader as parquet_reader
from cudf.bindings.parquet cimport reader_options as parquet_reader_options
from libc.stdlib cimport free
from libcpp.vector cimport vector
from libcpp.memory cimport unique_ptr

from cudf.dataframe.column import Column
from cudf.dataframe.dataframe import DataFrame
from cudf.utils import ioutils
from cudf.bindings.nvtx import nvtx_range_push, nvtx_range_pop

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

    # Setup reader options
    cdef parquet_reader_options options = parquet_reader_options()
    for col in columns or []:
        options.columns.push_back(str(col).encode())
    options.strings_to_categorical = strings_to_categorical

    # Create reader from source
    if not os.path.isfile(path) or not os.path.exists(path):
        raise FileNotFoundError(
            errno.ENOENT, os.strerror(errno.ENOENT), path
        )
    cdef unique_ptr[parquet_reader] reader
    reader = unique_ptr[parquet_reader](
        new parquet_reader(str(path).encode(), options)
    )

    # Read data into columns
    cdef cudf_table table
    if skip_rows is not None:
        table = reader.get().read_rows(
            skip_rows,
            num_rows if num_rows is not None else 0
        )
    elif num_rows is not None:
        table = reader.get().read_rows(
            skip_rows if skip_rows is not None else 0,
            num_rows
        )
    elif row_group is not None:
        table = reader.get().read_row_group(row_group)
    else:
        table = reader.get().read_all()

    # Extract read columns
    outcols = []
    new_names = []
    cdef gdf_column* column
    for i in range(table.num_columns()):
        column = table.get_column(i)
        data_mem, mask_mem = gdf_column_to_column_mem(column)
        outcols.append(Column.from_mem_views(data_mem, mask_mem))
        new_names.append(column.col_name.decode())
        free(column.col_name)
        free(column)

    # Construct dataframe from columns
    df = DataFrame()
    for k, v in zip(new_names, outcols):
        df[k] = v

    # Set column to use as row indexes if available
    index_col = reader.get().get_index_column().decode()
    if index_col is not '' and index_col in df.columns:
        df = df.set_index(index_col)
        new_index_name = pa.pandas_compat._backwards_compatible_index_name(
            df.index.name, df.index.name
        )
        df.index.name = new_index_name

    return df
