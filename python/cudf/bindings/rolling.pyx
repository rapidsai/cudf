from libc.stdlib cimport calloc, malloc, free

from cudf.dataframe.column import Column

from cudf.bindings.cudf_cpp cimport *
from cudf.bindings.cudf_cpp import *
from cudf.bindings.rolling cimport *


def apply_rolling(inp, window, min_periods, op):

    cdef gdf_column *inp_col

    if op == "mean":
        inp_col = column_view_from_column(inp.astype("float64"))
    else:
        inp_col = column_view_from_column(inp)

    cdef gdf_column *output_col = <gdf_column*> malloc(sizeof(gdf_column*))

    cdef gdf_index_type c_window = window
    cdef gdf_index_type c_min_periods = min_periods
    cdef gdf_agg_op c_op = agg_ops[op]

    with nogil:
        output_col = rolling_window(inp_col[0],
                                    c_window,
                                    c_min_periods,
                                    0,
                                    c_op,
                                    NULL,
                                    NULL,
                                    NULL
        )

    data, mask = gdf_column_to_column_mem(output_col)
    return Column.from_mem_views(data, mask)
