# Copyright (c) 2020, NVIDIA CORPORATION.

from __future__ import print_function
import cudf
import pandas as pd
import numba

from cudf._libxx.lib cimport *
from cudf._libxx.column cimport *

from cudf._libxx.aggregation cimport *

from cudf._libxx.includes.rolling cimport (
    rolling_window as cpp_rolling_window
)


def rolling(Column source_column, window, min_periods, center, op):
    """
    Rolling on input executing operation within the given window for each row

    Parameters
    ----------
    source_column : input column on which rolling operation is executed
    window : Size of the moving window, can be integer or numba DeviceArray
    min_periods : Minimum number of observations in window required to have
                  a value (otherwise result is null)
    center : Set the labels at the center of the window
    op : operation to be executed, as of now it supports MIN, MAX, COUNT, SUM,
         MEAN and UDF

    Returns
    -------
    A Column with rolling calculations
    """

    cdef size_type c_min_periods = min_periods
    cdef size_type c_window = 0
    cdef size_type c_forward_window = 0
    cdef unique_ptr[column] c_result

    if isinstance(window, numba.cuda.cudadrv.devicearray.DeviceNDArray):
        if center:
            # TODO: we can support this even though Pandas currently does not
            raise NotImplementedError(
                "center is not implemented for offset-based windows"
            )
        column_window = cudf.Series(window)._column
        f_column_window = cudf.Series([0] * column_window.size,
                                      dtype=column_window.dtype)._column

        pre_column_window = Column(
            column_window.data, column_window.size,
            column_window.dtype)

        fwd_column_window = Column(
            f_column_window.data, f_column_window.size,
            f_column_window.dtype)

        c_result = move(
            cpp_rolling_window(source_column.view(),
                               pre_column_window.view(),
                               fwd_column_window.view(),
                               c_min_periods,
                               get_aggregation(op,
                                               {'dtype': source_column.dtype}))
        )
    else:
        if op == "count":
            min_periods = 0
        c_min_periods = min_periods
        if center:
            c_window = (window // 2) + 1
            c_forward_window = window - (c_window)
        else:
            c_window = window
            c_forward_window = 0

        c_result = move(
            cpp_rolling_window(source_column.view(),
                               c_window,
                               c_forward_window,
                               c_min_periods,
                               get_aggregation(op,
                                               {'dtype': source_column.dtype}))
        )

    return Column.from_unique_ptr(move(c_result))
