# Copyright (c) 2024, NVIDIA CORPORATION.
from libc.stdint cimport int32_t
from libcpp.memory cimport unique_ptr
from libcpp.utility cimport move

from cudf._lib.pylibcudf.libcudf.column.column cimport column
from cudf._lib.pylibcudf.libcudf.round cimport (
    round as cpp_round,
    rounding_method,
)

from cudf._lib.pylibcudf.libcudf.round import rounding_method as RoundingMethod  # no-cython-lint, isort:skip

from .column cimport Column


cpdef Column round(
    Column values,
    int32_t decimal_places,
    rounding_method method
):
    """
    Round the input column to the specified number of decimal places.

    Parameters
    ----------
    values : Column
        The values to round.
    decimal_places : int
        Number of decimal places to round to.
    method : RoundingMethod
        Tie-breaking method for half.

    Returns
    -------
    Column
        Column with values rounded to the specified number of decimals.
    """
    cdef unique_ptr[column] result
    with nogil:
        result = move(
            cpp_round(values.view(), decimal_places, method)
        )
    return Column.from_libcudf(move(result))
