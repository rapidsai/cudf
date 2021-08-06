# Copyright (c) 2020-2021, NVIDIA CORPORATION.

import cudf

from libcpp cimport bool, int
from libcpp.memory cimport make_unique, unique_ptr
from libcpp.string cimport string
from libcpp.utility cimport move
from libcpp.vector cimport vector

from cudf._lib.cpp.column.column cimport column

from cudf.utils.dtypes import is_struct_dtype

from cudf._lib.column cimport Column
from cudf._lib.cpp.io.text cimport (
    read_text as libcudf_read_text,
    text_reader_options,
)
from cudf._lib.cpp.io.types cimport (
    column_name_info,
    compression_type,
    data_sink,
    sink_info,
    source_info,
    table_metadata,
    table_metadata_with_nullability,
    table_with_metadata,
)
from cudf._lib.cpp.table.table_view cimport table_view
from cudf._lib.cpp.types cimport data_type, size_type, type_id
from cudf._lib.io.utils cimport (
    make_sink_info,
    make_source_info,
    update_struct_field_names,
)
from cudf._lib.table cimport Table

from cudf._lib.types import np_to_cudf_types

from cudf._lib.types cimport underlying_type_t_type_id

import numpy as np

from cudf._lib.utils cimport get_column_names

from cudf._lib.utils import _index_level_name, generate_pandas_metadata


cpdef read_text(object filepaths_or_buffers,
                object delimiter=None,
                object compression=None):
    """
    Cython function to call into libcudf API, see `read_text`.

    See Also
    --------
    cudf.io.text.read_text
    """
    cdef text_reader_options c_text_reader_options = make_text_reader_options(
        filepaths_or_buffers,
        delimiter,
    )

    cdef table_with_metadata c_result
    with nogil:
        c_result = move(libcudf_read_text(c_text_reader_options))

    meta_names = [name.decode() for name in c_result.metadata.column_names]
    df = cudf.DataFrame._from_table(Table.from_unique_ptr(
        move(c_result.tbl),
        column_names=meta_names
    ))

    return df


cdef text_reader_options make_text_reader_options(
    object filepaths_or_buffers,
    object delimiter
) except*:

    cdef text_reader_options opts
    cdef source_info src = make_source_info(filepaths_or_buffers)

    opts = move(
        text_reader_options.builder(src)
        .delimiter(delimiter.encode())
        .build()
    )

    return opts
