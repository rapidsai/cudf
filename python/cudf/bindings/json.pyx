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

    cdef vector[const char*] vector_dtypes
    vector_dtypes = arr_dtypes

    # Setup arguments
    cdef json_read_arg args = json_read_arg()

    if is_file_like(path_or_buf):
        if compression == 'infer':
            compression = None
        source = path_or_buf.read()
        # check if StringIO is used
        if hasattr(source, 'encode'):
            source_as_bytes = source.encode()
        else:
            source_as_bytes = source
    else:
        # file path or a string
        source_as_bytes = str(path_or_buf).encode()

    source_data_holder = <char*>source_as_bytes
    args.source = source_data_holder

    if not is_file_like(path_or_buf) and os.path.exists(path_or_buf):
        if not os.path.isfile(path_or_buf):
            raise(FileNotFoundError)
        args.source_type = FILE_PATH
    else:
        args.source_type = HOST_BUFFER
        args.buffer_size = len(source_as_bytes)

    if compression is None or compression == 'infer':
        compression_bytes = <char*>NULL
    else:
        compression = compression.encode()
        compression_bytes = <char*>compression

    args.lines = lines
    args.compression = compression_bytes

    if dtype is not None:
        args.dtype = vector_dtypes.data()
        args.num_cols = vector_dtypes.size()
    else:
        args.dtype = NULL
        args.num_cols = 0

    if byte_range is not None:
        args.byte_range_offset = byte_range[0]
        args.byte_range_size = byte_range[1]
    else:
        args.byte_range_offset = 0
        args.byte_range_size = 0

    with nogil:
        result = read_json(&args)
    check_gdf_error(result)

    out = args.data
    if out is NULL:
        raise ValueError("Failed to parse Json")

    # Extract parsed columns
    outcols = []
    new_names = []
    for i in range(args.num_cols_out):
        data_mem, mask_mem = gdf_column_to_column_mem(out[i])
        outcols.append(Column.from_mem_views(data_mem, mask_mem))
        new_names.append(out[i].col_name.decode())
        free(out[i].col_name)
        free(out[i])

    # Construct dataframe from columns
    df = DataFrame()
    for k, v in zip(new_names, outcols):
        df[k] = v

    return df
