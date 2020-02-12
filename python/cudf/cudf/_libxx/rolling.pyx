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

def rolling(Column source_column, window, min_periods, center,  op):

    cdef size_type c_min_periods = min_periods
    cdef size_type c_window = 0
    cdef size_type c_forward_window = 0
    cdef unique_ptr[column] c_result
    
    if (isinstance(window, (Column, 
                            numba.cuda.cudadrv.devicearray.DeviceNDArray,
                            cudf.Series)) or pd.api.types.is_list_like(window)):
        if center:
            # TODO: we can support this even though Pandas currently does not
            raise NotImplementedError(
                "center is not implemented for offset-based windows"
            )
        column_window = None
        if isinstance(window, cudf.Series):
            column_window = window._column
        elif isinstance(window, Column):
            column_window = window
        else:
            column_window = cudf.Series(window)._column
            
        f_column_window  = cudf.Series([0]*column_window.size, dtype = column_window.dtype)._column
        pre_column_window = Column(column_window.data, column_window.size,
                               column_window.dtype, column_window.mask,
                               column_window.offset,
                               column_window.children)

        fwd_column_window = Column(f_column_window.data, f_column_window.size,
                               f_column_window.dtype)

        c_result = move(
                cpp_rolling_window(source_column.view(),
                                   pre_column_window.view(),
                                   fwd_column_window.view(),
                                   c_min_periods,
                                   get_aggregation(op, {'dtype': source_column.dtype}))
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
                                   get_aggregation(op, {'dtype': source_column.dtype}))
        )

    return Column.from_unique_ptr(move(c_result))
