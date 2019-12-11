# Copyright (c) 2019, NVIDIA CORPORATION.

# cython: profile=False
# distutils: language = c++
# cython: embedsignature = True
# cython: language_level = 3

from cudf._lib.cudf cimport *
from cudf._lib.cudf import *
from cudf._lib.includes.orc cimport (
    reader as orc_reader,
    reader_options as orc_reader_options,
    writer as orc_writer,
    writer_options as orc_writer_options,
    compression_type
)
from cython.operator cimport dereference as deref
from libc.stdlib cimport free
from libcpp.memory cimport unique_ptr, make_unique
from libcpp.string cimport string

from cudf._lib.utils cimport *
from cudf._lib.utils import *

import errno
import os


cdef unique_ptr[cudf_table] make_table_from_columns(columns):
    """
    Cython function to create a `cudf_table` from an ordered dict of columns
    """
    cdef vector[gdf_column*] c_columns
    for idx, (col_name, col) in enumerate(columns.items()):
        # Workaround for string columns
        if col.dtype.type == np.object_:
            c_columns.push_back(
                column_view_from_string_column(col, col_name)
            )
        else:
            c_columns.push_back(
                column_view_from_column(col, col_name)
            )

    return make_unique[cudf_table](c_columns)


cpdef read_orc(filepath_or_buffer, columns=None, stripe=None,
               skip_rows=None, num_rows=None, use_index=True,
               decimals_as_float=True, force_decimal_scale=None):
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
    options.decimals_as_float = decimals_as_float
    if force_decimal_scale is not None:
        options.forced_decimals_scale = force_decimal_scale

    # Create reader from source
    cdef const unsigned char[::1] buffer = view_of_buffer(filepath_or_buffer)
    cdef string filepath
    if buffer is None:
        if not os.path.isfile(filepath_or_buffer):
            raise FileNotFoundError(
                errno.ENOENT, os.strerror(errno.ENOENT), filepath_or_buffer
            )
        filepath = <string>str(filepath_or_buffer).encode()

    cdef unique_ptr[orc_reader] reader
    with nogil:
        if buffer is None:
            reader = unique_ptr[orc_reader](
                new orc_reader(filepath, options)
            )
        else:
            reader = unique_ptr[orc_reader](
                new orc_reader(<char *>&buffer[0], buffer.shape[0], options)
            )

    # Read data into columns
    cdef cudf_table c_out_table
    cdef size_type c_skip_rows = skip_rows if skip_rows is not None else 0
    cdef size_type c_num_rows = num_rows if num_rows is not None else -1
    cdef size_type c_stripe = stripe if stripe is not None else -1
    with nogil:
        if c_skip_rows != 0 or c_num_rows != -1:
            c_out_table = reader.get().read_rows(c_skip_rows, c_num_rows)
        elif c_stripe != -1:
            c_out_table = reader.get().read_stripe(c_stripe)
        else:
            c_out_table = reader.get().read_all()

    return table_to_dataframe(&c_out_table)


cpdef write_orc(cols, filepath_or_buffer, compression=None):
    """
    Cython function to call into libcudf API, see `write_orc`.

    See Also
    --------
    cudf.io.orc.read_orc
    """

    # Setup writer options
    cdef orc_writer_options options = orc_writer_options()
    if compression is None:
        options.compression = compression_type.none
    elif compression == "snappy":
        options.compression = compression_type.snappy
    else:
        raise ValueError("Unsupported `compression` type")

    # Create writer
    cdef string filepath = <string>str(filepath_or_buffer).encode()
    cdef unique_ptr[orc_writer] writer
    with nogil:
        writer = unique_ptr[orc_writer](
            new orc_writer(filepath, options)
        )

    # Write data to output
    cdef unique_ptr[cudf_table] c_in_table = make_table_from_columns(cols)
    with nogil:
        writer.get().write_all(deref(c_in_table))
