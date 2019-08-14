# Copyright (c) 2019, NVIDIA CORPORATION.

# cython: profile=False
# distutils: language = c++
# cython: embedsignature = True
# cython: language_level = 3

from cudf.bindings.cudf_cpp cimport *
from cudf.bindings.cudf_cpp import *
from cudf.bindings.orc cimport reader as orc_reader
from cudf.bindings.orc cimport reader_options as orc_reader_options
from libc.stdlib cimport free
from libcpp.memory cimport unique_ptr

from cudf.dataframe.column import Column
from cudf.dataframe.dataframe import DataFrame
from cudf.utils import ioutils
from cudf.bindings.nvtx import nvtx_range_push, nvtx_range_pop
from cudf.bindings.utils cimport *
from cudf.bindings.utils import *

from io import BytesIO
import errno
import os


cpdef cpp_read_orc(filepath_or_buffer, columns=None, stripe=None,
                   skip_rows=None, num_rows=None, use_index=True):
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
    cdef unique_ptr[orc_reader] reader
    cdef const unsigned char[:] buffer = None
    if isinstance(filepath_or_buffer, BytesIO):
        buffer = filepath_or_buffer.getbuffer()
    elif isinstance(filepath_or_buffer, bytes):
        buffer = filepath_or_buffer

    if buffer is not None:
        reader = unique_ptr[orc_reader](
            new orc_reader(<char *>&buffer[0], buffer.shape[0], options)
        )
    else:
        if not os.path.isfile(filepath_or_buffer):
            raise FileNotFoundError(
                errno.ENOENT, os.strerror(errno.ENOENT), filepath_or_buffer
            )
        reader = unique_ptr[orc_reader](
            new orc_reader(str(filepath_or_buffer).encode(), options)
        )

    # Read data into columns
    cdef cudf_table c_out_table
    if skip_rows is not None:
        c_out_table = reader.get().read_rows(
            skip_rows,
            num_rows if num_rows is not None else 0
        )
    elif num_rows is not None:
        c_out_table = reader.get().read_rows(
            skip_rows if skip_rows is not None else 0,
            num_rows
        )
    elif stripe is not None:
        c_out_table = reader.get().read_stripe(stripe)
    else:
        c_out_table = reader.get().read_all()

    return table_to_dataframe(&c_out_table)
