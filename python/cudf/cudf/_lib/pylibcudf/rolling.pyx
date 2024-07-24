# Copyright (c) 2024, NVIDIA CORPORATION.

from cython.operator cimport dereference
from libcpp cimport bool
from libcpp.memory cimport unique_ptr
from libcpp.utility cimport move, pair

from cudf._lib.pylibcudf.libcudf cimport rolling as cpp_rolling
from cudf._lib.pylibcudf.libcudf.aggregation cimport rolling_aggregation
from cudf._lib.pylibcudf.libcudf.column.column cimport column
from cudf._lib.pylibcudf.libcudf.rolling cimport window_type
from cudf._lib.pylibcudf.libcudf.types cimport size_type

from .aggregation cimport Aggregation
from .column cimport Column
from .scalar cimport Scalar

from cudf._lib.pylibcudf.libcudf.rolling import window_type as WindowType  # no-cython-lint, isort: skip


cpdef Column rolling_window(
    Column source,
    WindowArg preceding_window,
    WindowArg following_window,
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
    if WindowArg is Column:
        with nogil:
            result = move(
                cpp_rolling.rolling_window(
                    source.view(),
                    preceding_window.view(),
                    following_window.view(),
                    min_periods,
                    dereference(c_agg),
                )
            )
    else:
        with nogil:
            result = move(
                cpp_rolling.rolling_window(
                    source.view(),
                    preceding_window,
                    following_window,
                    min_periods,
                    dereference(c_agg),
                )
            )
    return Column.from_libcudf(move(result))


cpdef tuple[Column, Column] windows_from_offset(
    Column input,
    Scalar length,
    Scalar offset,
    window_type typ,
    bool only_preceding,
):
    """Compute rolling window bounds from a length and offset pair.

    For details, see :cpp:func:`windows_from_offset`.

    Parameters
    ----------
    input : Column
        Column that will be rolled over to define the windows.
    length : Scalar
        Length of the window at each element.
    offset : Scalar
        Offset to start of the window at each element.
    typ : WindowType
        Type of window, indicating which endpoints are contained.
    only_preceding : bool
        If true, avoid calculating the following window column.

    Returns
    -------
    tuple[Column, Column]
        A two-tuple of the preceding and following window columns
        suitable for passing to :func:`rolling_window`.
        If `only_preceding` is true, then the second entry is ``None``.
    """
    cdef pair[unique_ptr[column], unique_ptr[column]] result
    with nogil:
        result = cpp_rolling.windows_from_offset(
                input.view(),
                dereference(length.c_obj),
                dereference(offset.c_obj),
                typ,
                only_preceding,
            )
    if only_preceding:
        return (Column.from_libcudf(move(result.first)), None)
    else:
        return (
            Column.from_libcudf(move(result.first)),
            Column.from_libcudf(move(result.second)),
        )
