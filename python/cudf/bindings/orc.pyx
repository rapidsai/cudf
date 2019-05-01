# Copyright (c) 2019, NVIDIA CORPORATION.

# cython: profile=False
# distutils: language = c++
# cython: embedsignature = True
# cython: language_level = 3

from .cudf_cpp cimport *
from .cudf_cpp import *
from cudf.bindings.orc cimport *
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
import os


def is_file_like(obj):
    if not (hasattr(obj, 'read') or hasattr(obj, 'write')):
        return False
    if not hasattr(obj, "__iter__"):
        return False
    return True


cpdef cpp_read_orc(path, columns=None, skip_rows=None, num_rows=None):
    """
    Cython function to call into libcudf API, see `read_orc`.

    See Also
    --------
    cudf.io.orc.read_orc
    """

    # Setup arguments
    cdef orc_read_arg orc_reader = orc_read_arg()

    if not os.path.isfile(path) and not os.path.exists(path):
        raise FileNotFoundError(
            errno.ENOENT, os.strerror(errno.ENOENT), path
        )
    path = str(path).encode()
    source_ptr = <char*>path
    orc_reader.source_type = FILE_PATH
    orc_reader.source = source_ptr

    usecols = columns
    cdef vector[const char*] vector_use_cols
    if usecols is not None:
        arr_cols = []
        for col in usecols:
            arr_cols.append(str(col).encode())
        vector_use_cols = arr_cols
        orc_reader.use_cols = vector_use_cols.data()
        orc_reader.use_cols_len = len(usecols)

    if skip_rows is not None:
        orc_reader.skip_rows = skip_rows
    if num_rows is not None:
        orc_reader.num_rows = num_rows
    else:
        orc_reader.num_rows = -1

    # Call read_orc
    with nogil:
        result = read_orc(&orc_reader)

    check_gdf_error(result)

    out = orc_reader.data
    if out is NULL:
        raise ValueError("Failed to parse ORC")

    # Extract parsed columns
    outcols = []
    new_names = []
    for i in range(orc_reader.num_cols_out):
        data_mem, mask_mem = gdf_column_to_column_mem(out[i])
        if (out[i].dtype_info.time_unit == TIME_UNIT_ns):
            newcol = Column.from_mem_views(data_mem, mask_mem)
            outcols.append(
                newcol.view(DatetimeColumn, dtype='datetime64[ns]')
            )
        else:
            outcols.append(Column.from_mem_views(data_mem, mask_mem))
        new_names.append(out[i].col_name.decode())
        free(out[i].col_name)
        free(out[i])

    # Construct dataframe from columns
    df = DataFrame()
    for k, v in zip(new_names, outcols):
        df[k] = v

    return df
