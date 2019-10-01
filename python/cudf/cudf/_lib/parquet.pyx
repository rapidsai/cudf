# Copyright (c) 2019, NVIDIA CORPORATION.

# cython: profile=False
# distutils: language = c++
# cython: embedsignature = True
# cython: language_level = 3

from cudf._lib.cudf cimport *
from cudf._lib.cudf import *
from cudf._lib.utils cimport *
from cudf._lib.utils import *
from cudf._lib.includes.parquet cimport (
    reader as parquet_reader,
    reader_options as parquet_reader_options
)
from libc.stdlib cimport free
from libcpp.memory cimport unique_ptr
from libcpp.string cimport string

import errno
import os


cpdef read_parquet(filepath_or_buffer, columns=None, row_group=None,
                   skip_rows=None, num_rows=None,
                   strings_to_categorical=False, use_pandas_metadata=False):
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
    options.use_pandas_metadata = use_pandas_metadata

    # Create reader from source
    cdef const unsigned char[::1] buffer = view_of_buffer(filepath_or_buffer)
    cdef string filepath
    if buffer is None:
        if not os.path.isfile(filepath_or_buffer):
            raise FileNotFoundError(
                errno.ENOENT, os.strerror(errno.ENOENT), filepath_or_buffer
            )
        filepath = <string>str(filepath_or_buffer).encode()

    cdef unique_ptr[parquet_reader] reader
    with nogil:
        if buffer is None:
            reader = unique_ptr[parquet_reader](
                new parquet_reader(filepath, options)
            )
        else:
            reader = unique_ptr[parquet_reader](
                new parquet_reader(<char*>&buffer[0], buffer.shape[0], options)
            )

    # Read data into columns
    cdef cudf_table c_out_table
    cdef int c_skip_rows = skip_rows if skip_rows is not None else 0
    cdef int c_num_rows = num_rows if num_rows is not None else -1
    cdef int c_row_group = row_group if row_group is not None else -1
    with nogil:
        if c_skip_rows != 0 or c_num_rows != -1:
            c_out_table = reader.get().read_rows(c_skip_rows, c_num_rows)
        elif c_row_group != -1:
            c_out_table = reader.get().read_row_group(c_row_group)
        else:
            c_out_table = reader.get().read_all()

    # Construct dataframe from columns
    df = table_to_dataframe(&c_out_table)

    # Set column to use as row indexes if available
    index_col = reader.get().get_index_column().decode("UTF-8")
    if index_col is not '' and index_col in df.columns:
        df = df.set_index(index_col)
        new_index_name = pa.pandas_compat._backwards_compatible_index_name(
            df.index.name, df.index.name
        )
        df.index.name = new_index_name

    return df
