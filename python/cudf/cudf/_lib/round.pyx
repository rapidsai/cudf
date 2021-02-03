# Copyright (c) 2021, NVIDIA CORPORATION.

from libcpp.memory cimport unique_ptr
from libcpp.utility cimport move

from cudf._lib.column cimport Column

from cudf._lib.cpp.column.column cimport column
from cudf._lib.cpp.column.column_view cimport column_view
from cudf._lib.cpp.round cimport (
    rounding_method as cpp_rounding_method,
    round as cpp_round
)


def round(Column input_col, int decimal_places=0):
    """
    Round column values to the given number of decimal places

    Parameters
    ----------
    input_col : Column whose values will be rounded
    decimal_places : The number or decimal places to round to

    Returns
    -------
    A Column with values rounded to the given number of decimal places
    """

    cdef column_view input_col_view = input_col.view()
    cdef unique_ptr[column] c_result

    with nogil:
        c_result = move(
            cpp_round(
                input_col_view,
                decimal_places,
                cpp_rounding_method.HALF_EVEN,
            )
        )

    return Column.from_unique_ptr(move(c_result))
