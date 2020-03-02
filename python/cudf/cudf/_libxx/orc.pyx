# Copyright (c) 2020, NVIDIA CORPORATION.

from libcpp cimport bool, int
from libcpp.memory cimport unique_ptr
from libcpp.string cimport string
from libcpp.vector cimport vector
from cudf._libxx.cpp.column.column cimport column

from cudf._libxx.cpp.io.functions cimport (
    read_orc_args,
    write_orc_args,
    read_orc as read_orc_cpp,
    write_orc as write_orc_cpp
)
from cudf._libxx.cpp.io.types cimport (
    compression_type,
    sink_info,
    table_metadata,
    table_with_metadata,
)
from cudf._libxx.cpp.types cimport (
    data_type, type_id, size_type
)

from cudf._libxx.io.utils cimport make_source_info
from cudf._libxx.move cimport move
from cudf._libxx.table cimport Table
from cudf._libxx.types import np_to_cudf_types
import numpy as np


cpdef read_orc(filepath_or_buffer, columns=None,
               stripe=None, stripe_count=None,
               skip_rows=None, num_rows=None, use_index=True,
               decimals_as_float=True, force_decimal_scale=None,
               timestamp_type=None):
    """
    Cython function to call into libcudf API, see `read_orc`.

    See Also
    --------
    cudf.io.orc.read_orc
    """
    cdef read_orc_args c_read_orc_args = make_read_orc_args(
        filepath_or_buffer,
        columns or [],
        size_t_arg(stripe, "stripe"),
        size_t_arg(stripe_count, "stripe_count"),
        size_t_arg(skip_rows, "skip_rows"),
        size_t_arg(num_rows, "num_rows"),
        (
            type_id.EMPTY
            if timestamp_type is None else
            np_to_cudf_types[np.dtype(timestamp_type)]
        ),
        use_index,
        decimals_as_float,
        size_t_arg(force_decimal_scale, "force_decimal_scale")
    )

    cdef table_with_metadata c_result

    with nogil:
        c_result = move(read_orc_cpp(c_read_orc_args))

    names = [name.decode() for name in c_result.metadata.column_names]

    return Table.from_unique_ptr(move(c_result.tbl), names)


cpdef write_orc(Table table,
                filepath,
                compression=None,
                enable_statistics=False):
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
    metadata_.column_names.reserve(len(table._column_names))

    for col_name in table._column_names:
        metadata_.column_names.push_back(str.encode(col_name))

    cdef write_orc_args c_write_orc_args = write_orc_args(
        sink_info(<string>str(filepath).encode()),
        table.data_view(), &metadata_, compression_,
        <bool> (True if enable_statistics else False)
    )

    with nogil:
        write_orc_cpp(c_write_orc_args)


cdef size_type size_t_arg(arg, name) except*:
    arg = -1 if arg is None else arg
    if not isinstance(arg, int) or arg < -1:
        raise TypeError("{} must be an int >= -1".format(name))
    return <size_type> arg


cdef read_orc_args make_read_orc_args(p, column_names,
                                      size_type stripe,
                                      size_type stripe_count,
                                      size_type skip_rows,
                                      size_type num_rows,
                                      type_id timestamp_type,
                                      bool use_index,
                                      bool decimals_as_float,
                                      size_type force_decimal_scale) except*:
    cdef read_orc_args args = read_orc_args(make_source_info(p))
    args.stripe = <size_type> stripe
    args.stripe_count = <size_type> stripe_count
    args.skip_rows = <size_type> skip_rows
    args.num_rows = <size_type> num_rows
    args.timestamp_type = data_type(timestamp_type)
    args.use_index = <bool> use_index
    args.decimals_as_float = <bool> decimals_as_float
    args.forced_decimals_scale = <int> force_decimal_scale
    args.columns.reserve(len(column_names))
    for col in column_names:
        args.columns.push_back(str(col).encode())
    return args
