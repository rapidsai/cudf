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
    writer_options as orc_writer_options
)
from cython.operator cimport dereference as deref
from libcpp.memory cimport unique_ptr, make_unique

from cudf._lib.utils cimport *
from cudf._lib.utils import *

from io import BytesIO
import errno
import os


cdef unique_ptr[cudf_table] make_table_from_columns(columns):
    """
    Cython function to create a `cudf_table` from an ordered dict of columns
    """
    cdef vector[gdf_column*] c_columns

    for idx, (col_name, col) in enumerate(columns.items()):
        check_gdf_compatibility(col._column)
        c_columns.push_back(column_view_from_column(col._column, col_name))

    return make_unique[cudf_table](c_columns)


cpdef read_orc(filepath_or_buffer, columns=None, stripe=None,
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


cpdef write_orc(cols, filepath):
    """
    Cython function to call into libcudf API, see `write_orc`.

    See Also
    --------
    cudf.io.orc.read_orc
    """

    # Setup writer options
    cdef orc_writer_options options = orc_writer_options()

    # Create writer
    cdef unique_ptr[orc_writer] writer
    writer = unique_ptr[orc_writer](
        new orc_writer(str(filepath).encode(), options)
    )

    # Write data to output
    cdef unique_ptr[cudf_table] c_in_table = make_table_from_columns(cols)
    writer.get().write_all(deref(c_in_table))
