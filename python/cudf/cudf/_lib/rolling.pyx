# Copyright (c) 2020-2022, NVIDIA CORPORATION.

from libcpp.memory cimport unique_ptr
from libcpp.utility cimport move

from cudf._lib.aggregation cimport RollingAggregation, make_rolling_aggregation
from cudf._lib.column cimport Column
from cudf._lib.cpp.column.column cimport column
from cudf._lib.cpp.column.column_view cimport column_view
from cudf._lib.cpp.rolling cimport rolling_window as cpp_rolling_window
from cudf._lib.cpp.types cimport size_type


def rolling(Column source_column,
            Column pre_column_window,
            Column fwd_column_window,
            window,
            min_periods,
            center,
            op,
            agg_params):
    """
    Rolling on input executing operation within the given window for each row

    Parameters
    ----------
    source_column : input column on which rolling operation is executed
    pre_column_window : prior window for each element of source_column
    fwd_column_window : forward window for each element of source_column
    window : Size of the moving window, can be integer or None
    min_periods : Minimum number of observations in window required to have
                  a value (otherwise result is null)
    center : Set the labels at the center of the window
    op : operation to be executed
    agg_params : dict, parameter for the aggregation (e.g. ddof for VAR/STD)

    Returns
    -------
    A Column with rolling calculations
    """
    cdef size_type c_min_periods = min_periods
    cdef size_type c_window = 0
    cdef size_type c_forward_window = 0
    cdef unique_ptr[column] c_result
    cdef column_view source_column_view = source_column.view()
    cdef column_view pre_column_window_view
    cdef column_view fwd_column_window_view
    cdef RollingAggregation cython_agg

    if callable(op):
        cython_agg = make_rolling_aggregation(
            op, {'dtype': source_column.dtype})
    else:
        cython_agg = make_rolling_aggregation(op, agg_params)

    if window is None:
        if center:
            # TODO: we can support this even though Pandas currently does not
            raise NotImplementedError(
                "center is not implemented for offset-based windows"
            )
        pre_column_window_view = pre_column_window.view()
        fwd_column_window_view = fwd_column_window.view()
        with nogil:
            c_result = move(
                cpp_rolling_window(
                    source_column_view,
                    pre_column_window_view,
                    fwd_column_window_view,
                    c_min_periods,
                    cython_agg.c_obj.get()[0])
            )
    else:
        c_min_periods = min_periods
        if center:
            c_window = (window // 2) + 1
            c_forward_window = window - (c_window)
        else:
            c_window = window
            c_forward_window = 0

        with nogil:
            c_result = move(
                cpp_rolling_window(
                    source_column_view,
                    c_window,
                    c_forward_window,
                    c_min_periods,
                    cython_agg.c_obj.get()[0])
            )

    return Column.from_unique_ptr(move(c_result))
