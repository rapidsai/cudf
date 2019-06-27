# Copyright (c) 2019, NVIDIA CORPORATION.

# cython: profile=False
# distutils: language = c++
# cython: embedsignature = True
# cython: language_level = 3

from .cudf_cpp cimport *
from .cudf_cpp import *
from cudf.bindings.orc cimport reader as orc_reader
from cudf.bindings.orc cimport reader_options as orc_reader_options
from libc.stdlib cimport free
from libcpp.memory cimport unique_ptr

from cudf.dataframe.column import Column
from cudf.dataframe.dataframe import DataFrame
from cudf.utils import ioutils
from cudf.bindings.nvtx import nvtx_range_push, nvtx_range_pop

import errno
import os


def is_file_like(obj):
    if not (hasattr(obj, 'read') or hasattr(obj, 'write')):
        return False
    if not hasattr(obj, "__iter__"):
        return False
    return True


cpdef cpp_read_orc(path, columns=None, stripe=None, skip_rows=None,
                   num_rows=None, use_index=True):
    """
    Cython function to call into libcudf API, see `read_orc`.

    See Also
    --------
    cudf.io.orc.read_orc
    """

    # Setup reader options
    cdef orc_reader_options options = orc_reader_options()
    for col in columns or []:
        options.columns.push_back(str(col).encode())
    options.use_index = use_index

    # Create reader from source
    if not os.path.isfile(path) or not os.path.exists(path):
        raise FileNotFoundError(
            errno.ENOENT, os.strerror(errno.ENOENT), path
        )
    cdef unique_ptr[orc_reader] reader
    reader = unique_ptr[orc_reader](new orc_reader(str(path).encode(), options))

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
    elif stripe is not None:
        table = reader.get().read_stripe(stripe)
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

    return df
