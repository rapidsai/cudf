# Copyright (c) 2024, NVIDIA CORPORATION.

from cython.operator cimport dereference
from libcpp.memory cimport unique_ptr
from libcpp.utility cimport move
from pylibcudf.libcudf cimport rolling as cpp_rolling
from pylibcudf.libcudf.aggregation cimport rolling_aggregation
from pylibcudf.libcudf.column.column cimport column
from pylibcudf.libcudf.types cimport size_type

from .aggregation cimport Aggregation
from .column cimport Column

__all__ = ["rolling_window"]

cpdef Column rolling_window(
    Column source,
    WindowType preceding_window,
    WindowType following_window,
    size_type min_periods,
    Aggregation agg,
):
    """Perform a rolling window operation on a column

    For details, see ``cudf::rolling_window`` documentation.

    Parameters
    ----------
    source : Column
        The column to perform the rolling window operation on.
    preceding_window : Union[Column, size_type]
        The column containing the preceding window sizes or a scalar value
        indicating the sizes of all windows.
    following_window : Union[Column, size_type]
        The column containing the following window sizes or a scalar value
        indicating the sizes of all windows.
    min_periods : int
        The minimum number of periods to include in the result.
    agg : Aggregation
        The aggregation to perform.

    Returns
    -------
    Column
        The result of the rolling window operation.
    """
    cdef unique_ptr[column] result
    # TODO: Consider making all the conversion functions nogil functions that
    # reclaim the GIL internally for just the necessary scope like column.view()
    cdef const rolling_aggregation *c_agg = agg.view_underlying_as_rolling()
    if WindowType is Column:
        with nogil:
            result = cpp_rolling.rolling_window(
                source.view(),
                preceding_window.view(),
                following_window.view(),
                min_periods,
                dereference(c_agg),
            )
    else:
        with nogil:
            result = cpp_rolling.rolling_window(
                source.view(),
                preceding_window,
                following_window,
                min_periods,
                dereference(c_agg),
            )

    return Column.from_libcudf(move(result))
