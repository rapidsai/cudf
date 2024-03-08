# Copyright (c) 2024, NVIDIA CORPORATION.
from libcpp.memory cimport unique_ptr
from libcpp.utility cimport move

from cudf._lib.cpp.column.column cimport column
from cudf._lib.cpp.column.column_factories cimport (
    make_duration_column as cpp_make_duration_column,
    make_empty_column as cpp_make_empty_column,
    make_fixed_point_column as cpp_make_fixed_point_column,
    make_numeric_column as cpp_make_numeric_column,
    make_timestamp_column as cpp_make_timestamp_column,
)
from cudf._lib.cpp.types cimport mask_state, size_type

from .types cimport DataType


cpdef Column make_empty_column(MakeEmptyColumnOperand type_or_id):
    cdef unique_ptr[column] result

    if MakeEmptyColumnOperand is DataType:
        with nogil:
            result = move(
                cpp_make_empty_column(
                    type_or_id.c_obj
                )
            )
    else:
        with nogil:
            result = move(
                cpp_make_empty_column(
                    type_or_id
                )
            )

    return Column.from_libcudf(move(result))


cpdef Column make_numeric_column(
    DataType type_,
    size_type size,
    mask_state state
):

    cdef unique_ptr[column] result

    with nogil:
        result = move(
            cpp_make_numeric_column(
                type_.c_obj,
                size,
                state
            )
        )
    return Column.from_libcudf(move(result))

cdef Column make_timestamp_column(
    DataType type_,
    size_type size,
    mask_state state
):
    cdef unique_ptr[column] result

    with nogil:
        result = move(
            cpp_make_timestamp_column(
                type_.c_obj,
                size,
                state
            )
        )
    return Column.from_libcudf(move(result))

cdef Column make_duration_column(
    DataType type_,
    size_type size,
    mask_state state
):
    cdef unique_ptr[column] result

    with nogil:
        result = move(
            cpp_make_duration_column(
                type_.c_obj,
                size,
                state
            )
        )
    return Column.from_libcudf(move(result))

cdef Column make_fixed_point_column(
    DataType type_,
    size_type size,
    mask_state state
):
    cdef unique_ptr[column] result

    with nogil:
        result = move(
            cpp_make_fixed_point_column(
                type_.c_obj,
                size,
                state
            )
        )
    return Column.from_libcudf(move(result))
