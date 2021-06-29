# Copyright (c) 2021, NVIDIA CORPORATION.

import numpy as np

from cudf._lib.column cimport Column
from cudf._lib.types import np_to_cudf_types
from cudf._lib.types cimport underlying_type_t_type_id
from cudf._lib.cpp.types cimport DECIMAL64

from cudf.core.column.column import as_column

from cudf._lib.cpp.column.column cimport column
from cudf._lib.cpp.column.column_view cimport column_view
from cudf._lib.cpp.strings.convert.convert_fixed_point cimport (
    to_fixed_point as cpp_to_fixed_point,
    from_fixed_point as cpp_from_fixed_point,
    is_fixed_point as cpp_is_fixed_point
)
from cudf._lib.cpp.types cimport (
    type_id,
    data_type,
)

from libcpp.memory cimport unique_ptr
from libcpp.utility cimport move
from libcpp.string cimport string


def from_decimal(Column input_col):
    """
    Converts a `Decimal64Column` to a `StringColumn`.

    Parameters
    ----------
    input_col : input column of type decimal

    Returns
    -------
    A column of strings representing the input decimal values.
    """
    cdef column_view input_column_view = input_col.view()
    cdef unique_ptr[column] c_result
    with nogil:
        c_result = move(
            cpp_from_fixed_point(
                input_column_view))

    return Column.from_unique_ptr(move(c_result))


def to_decimal(Column input_col, object out_type):
    """
    Returns a `Decimal64Column` from the provided `StringColumn`
    using the scale in the `out_type`.

    Parameters
    ----------
    input_col : input column of type string
    out_type : The type and scale of the decimal column expected

    Returns
    -------
    A column of decimals parsed from the string values.
    """
    cdef column_view input_column_view = input_col.view()
    cdef unique_ptr[column] c_result
    cdef int scale = out_type.scale
    cdef data_type c_out_type = data_type(DECIMAL64, -scale)
    with nogil:
        c_result = move(
            cpp_to_fixed_point(
                input_column_view,
                c_out_type))

    result = Column.from_unique_ptr(move(c_result))
    result.dtype.precision = out_type.precision
    return result


def is_fixed_point(Column input_col, object dtype):
    """
    Returns a Column of boolean values with True for `input_col`
    that have fixed-point characters. The output row also has a
    False value if the corresponding string would cause an integer
    overflow. The scale of the `dtype` is used to determine overflow
    in the output row.

    Parameters
    ----------
    input_col : input column of type string
    dtype : The type and scale of a decimal column

    Returns
    -------
    A Column of booleans indicating valid decimal conversion.
    """
    cdef unique_ptr[column] c_result
    cdef column_view source_view = input_col.view()
    cdef int scale = dtype.scale
    cdef data_type c_dtype = data_type(DECIMAL64, -scale)
    with nogil:
        c_result = move(cpp_is_fixed_point(
            source_view,
            c_dtype
        ))

    return Column.from_unique_ptr(move(c_result))
