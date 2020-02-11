# Copyright (c) 2020, NVIDIA CORPORATION.

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
    
    if (isinstance(window, Column) or isinstance(window, cudf.Series) or
        pd.api.types.is_list_like(window)):
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
            column_window = cudf.Series(window)._columnn
            
        tmp_column_window = Column(column_window.data, column_window.size,
                               column_window.dtype, column_window.mask,
                               column_window.offset,
                               column_window.children)

        c_result = move(
                cpp_rolling_window(source_column.view(),
                                   tmp_column_window.view(),
                                   tmp_column_window.view(),
                                   c_min_periods,
                                   get_aggregation(op, {'dtype': source_column.dtype}))
        )
    else:
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
