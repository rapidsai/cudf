# Copyright (c) 2020, NVIDIA CORPORATION.

from libcpp cimport bool, int
from libcpp.memory cimport unique_ptr
from libcpp.string cimport string
from libcpp.vector cimport vector
from cudf._lib.cpp.column.column cimport column

from cudf._lib.cpp.io.orc_metadata cimport (
    read_orc_statistics as libcudf_read_orc_statistics
)
from cudf._lib.cpp.io.orc cimport (
    orc_reader_options,
    read_orc as libcudf_read_orc,
    orc_writer_options,
    write_orc as libcudf_write_orc,
)
from cudf._lib.cpp.io.types cimport (
    compression_type,
    sink_info,
    source_info,
    table_metadata,
    table_with_metadata,
    data_sink,
)
from cudf._lib.cpp.types cimport (
    data_type, type_id, size_type
)

from cudf._lib.io.utils cimport make_source_info, make_sink_info
from cudf._lib.move cimport move
from cudf._lib.table cimport Table
from cudf._lib.types import np_to_cudf_types
from cudf._lib.types cimport underlying_type_t_type_id
import numpy as np


cpdef read_orc_statistics(filepath_or_buffer):
    """
    Cython function to call into libcudf API, see `read_orc_statistics`.

    See Also
    --------
    cudf.io.orc.read_orc_statistics
    """
    return libcudf_read_orc_statistics(make_source_info([filepath_or_buffer]))


cpdef read_orc(filepath_or_buffer, columns=None, stripes=None,
               skip_rows=None, num_rows=None, use_index=True,
               decimals_as_float=True, force_decimal_scale=None,
               timestamp_type=None):
    """
    Cython function to call into libcudf API, see `read_orc`.

    See Also
    --------
    cudf.io.orc.read_orc
    """
    cdef orc_reader_options c_orc_reader_options = make_orc_reader_options(
        filepath_or_buffer,
        columns or [],
        stripes or [],
        get_size_t_arg(skip_rows, "skip_rows"),
        get_size_t_arg(num_rows, "num_rows"),
        (
            type_id.EMPTY
            if timestamp_type is None else
            <type_id>(
                <underlying_type_t_type_id> (
                    np_to_cudf_types[np.dtype(timestamp_type)]
                )
            )
        ),
        use_index,
        decimals_as_float,
        get_size_t_arg(force_decimal_scale, "force_decimal_scale")
    )

    cdef table_with_metadata c_result

    with nogil:
        c_result = move(libcudf_read_orc(c_orc_reader_options))

    names = [name.decode() for name in c_result.metadata.column_names]

    return Table.from_unique_ptr(move(c_result.tbl), names)


cpdef write_orc(Table table,
                path_or_buf,
                compression=None,
                enable_statistics=True):
    """
    Cython function to call into libcudf API, see `write_orc`.

    See Also
    --------
    cudf.io.orc.read_orc
    """
    cdef compression_type compression_ = compression_type.NONE
    if compression is None or compression is False:
        compression_ = compression_type.NONE
    elif compression == "snappy":
        compression_ = compression_type.SNAPPY
    else:
        raise ValueError(
            "Unsupported compression type `{}`".format(compression)
        )

    cdef table_metadata metadata_ = table_metadata()
    cdef unique_ptr[data_sink] data_sink_c
    cdef sink_info sink_info_c = make_sink_info(path_or_buf, &data_sink_c)

    metadata_.column_names.reserve(len(table._column_names))

    for col_name in table._column_names:
        metadata_.column_names.push_back(str.encode(col_name))

    cdef orc_writer_options c_orc_writer_options = move(
        orc_writer_options.builder(sink_info_c, table.data_view()).
        metadata(&metadata_).
        compression(compression_).
        enable_statistics(<bool> (True if enable_statistics else False)).
        build()
    )

    with nogil:
        libcudf_write_orc(c_orc_writer_options)


cdef size_type get_size_t_arg(arg, name) except*:
    if name == "skip_rows":
        arg = 0 if arg is None else arg
        if not isinstance(arg, int) or arg < 0:
            raise TypeError("{} must be an int >= 0".format(name))
    else:
        arg = -1 if arg is None else arg
        if not isinstance(arg, int) or arg < -1:
            raise TypeError("{} must be an int >= -1".format(name))
    return <size_type> arg


cdef orc_reader_options make_orc_reader_options(
    filepath_or_buffer,
    column_names,
    stripes,
    size_type skip_rows,
    size_type num_rows,
    type_id timestamp_type,
    bool use_index,
    bool decimals_as_float,
    size_type force_decimal_scale
) except*:

    cdef vector[string] c_column_names
    cdef vector[size_type] strps = stripes
    c_column_names.reserve(len(column_names))
    for col in column_names:
        c_column_names.push_back(str(col).encode())
    cdef orc_reader_options opts
    cdef source_info src = make_source_info([filepath_or_buffer])
    opts = move(
        orc_reader_options.builder(src).
        columns(c_column_names).
        stripes(strps).
        skip_rows(skip_rows).
        num_rows(num_rows).
        timestamp_type(data_type(timestamp_type)).
        use_index(use_index).
        decimals_as_float(decimals_as_float).
        forced_decimals_scale(force_decimal_scale).
        build()
    )

    return opts
