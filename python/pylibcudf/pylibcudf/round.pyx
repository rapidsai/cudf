# Copyright (c) 2024-2025, NVIDIA CORPORATION.
from libc.stdint cimport int32_t
from libcpp.memory cimport unique_ptr
from libcpp.utility cimport move
from pylibcudf.libcudf.round cimport round as cpp_round, rounding_method

from pylibcudf.libcudf.round import \
    rounding_method as RoundingMethod  # no-cython-lint

from pylibcudf.libcudf.column.column cimport column

from rmm.pylibrmm.stream cimport Stream

from .column cimport Column
from .utils cimport _get_stream

__all__ = ["RoundingMethod", "round"]

cpdef Column round(
    Column source,
    int32_t decimal_places = 0,
    rounding_method round_method = rounding_method.HALF_UP,
    Stream stream=None
):
    """Rounds all the values in a column to the specified number of decimal places.

    For details, see :cpp:func:`round`.

    Parameters
    ----------
    source : Column
        The Column for which to round values.
    decimal_places: int32_t, optional
        The number of decimal places to round to (default 0)
    round_method: rounding_method, optional
        The method by which to round each value.
        Can be one of { RoundingMethod.HALF_UP, RoundingMethod.HALF_EVEN }
        (default rounding_method.HALF_UP)
    stream : Stream | None
        CUDA stream on which to perform the operation.

    Returns
    -------
    pylibcudf.Column
        A Column with values rounded
    """
    cdef unique_ptr[column] c_result
    stream = _get_stream(stream)

    with nogil:
        c_result = cpp_round(
            source.view(),
            decimal_places,
            round_method,
            stream.view()
        )

    return Column.from_libcudf(move(c_result), stream)

RoundingMethod.__str__ = RoundingMethod.__repr__
