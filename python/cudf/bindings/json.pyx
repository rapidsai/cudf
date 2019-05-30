# Copyright (c) 2019, NVIDIA CORPORATION.

# cython: profile=False
# distutils: language = c++
# cython: embedsignature = True
# cython: language_level = 3

from cudf.bindings.cudf_cpp cimport *
from cudf.bindings.cudf_cpp import *
from cudf.bindings.json cimport *
from libc.stdlib cimport free
from libcpp.vector cimport vector
from libcpp.string cimport string

from cudf.bindings.types cimport table as cudf_table
from cudf.dataframe.dataframe import DataFrame
from cudf.dataframe.column import Column
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

cpdef cpp_read_json(path_or_buf, dtype, lines, compression, byte_range):
    """
    Cython function to call into libcudf API, see `read_json`.
    See Also
    --------
    cudf.io.json.read_json
    cudf.io.json.to_json
    """

    if dtype is False:
        raise ValueError("cudf engine does not support dtype==False. "
                         "Pass True to enable data type inference, or "
                         "pass a list/dict of types to specify them manually.")
    arr_dtypes = []
    if dtype is not True:
        if isinstance(dtype, collections.abc.Mapping):
            for col, dt in dtype.items():
                arr_dtypes.append(str(str(col) + ":" + str(dt)).encode())
        elif not isinstance(dtype, collections.abc.Iterable):
            msg = '''dtype must be 'list like' or 'dict' '''
            raise TypeError(msg)
        else:
            for dt in dtype:
                arr_dtypes.append(dt.encode())

    # Setup arguments
    cdef json_read_arg args = json_read_arg()

    if is_file_like(path_or_buf):
        source = path_or_buf.read()
        # check if StringIO is used
        if hasattr(source, 'encode'):
            args.source = source.encode()
        else:
            args.source = source
    else:
        # file path or a string
        args.source = str(path_or_buf).encode()

    if not is_file_like(path_or_buf) and os.path.exists(path_or_buf):
        if not os.path.isfile(path_or_buf):
            raise(FileNotFoundError)
        args.source_type = FILE_PATH
    else:
        args.source_type = HOST_BUFFER

    if compression is None:
        args.compression = b'none'
    else:
        args.compression = compression.encode()

    args.lines = lines

    if dtype is not None:
        args.dtype = arr_dtypes

    if byte_range is not None:
        args.byte_range_offset = byte_range[0]
        args.byte_range_size = byte_range[1]
    else:
        args.byte_range_offset = 0
        args.byte_range_size = 0

    cdef cudf_table table
    with nogil:
        table = read_json(args)

    # Extract parsed columns
    outcols = []
    new_names = []
    for i in range(table.num_columns()):
        data_mem, mask_mem = gdf_column_to_column_mem(table.get_column(i))
        outcols.append(Column.from_mem_views(data_mem, mask_mem))
        new_names.append(table.get_column(i).col_name.decode())
        free(table.get_column(i).col_name)
        free(table.get_column(i))

    # Construct dataframe from columns
    df = DataFrame()
    for k, v in zip(new_names, outcols):
        df[k] = v

    return df
