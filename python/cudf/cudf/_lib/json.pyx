# Copyright (c) 2019, NVIDIA CORPORATION.

# cython: profile=False
# distutils: language = c++
# cython: embedsignature = True
# cython: language_level = 3


from cudf._lib.cudf cimport *
from cudf._lib.cudf import *
from cudf._lib.includes.json cimport (
    reader as json_reader,
    reader_options as json_reader_options
)
from cudf._lib.includes.io cimport FILE_PATH, HOST_BUFFER

from libc.stdlib cimport free
from libcpp.vector cimport vector
from libcpp.string cimport string
from libcpp.memory cimport unique_ptr

from cudf._lib.utils cimport *
from cudf._lib.utils import *

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


cpdef read_json(path_or_buf, dtype, lines, compression, byte_range):
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
    cdef json_reader_options args = json_reader_options()

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

    cdef unique_ptr[json_reader] reader
    with nogil:
        reader = unique_ptr[json_reader](new json_reader(args))

    cdef cudf_table c_out_table
    if byte_range is None:
        c_out_table = reader.get().read()
    else:
        c_out_table = reader.get().read_byte_range(
            byte_range[0], byte_range[1]
        )

    return table_to_dataframe(&c_out_table)
