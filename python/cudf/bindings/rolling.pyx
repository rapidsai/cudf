from libc.stdlib cimport calloc, malloc, free

from cudf.dataframe.column import Column

from cudf.bindings.cudf_cpp cimport *
from cudf.bindings.cudf_cpp import *
from cudf.bindings.rolling cimport *


def apply_rolling(inp, window, min_periods, op):

    cdef gdf_column *inp_col
    cdef gdf_column *output_col = <gdf_column*> malloc(sizeof(gdf_column*))
    cdef gdf_index_type c_window = window
    cdef gdf_agg_op c_op = agg_ops[op]

    if op == "mean":
        inp_col = column_view_from_column(inp.astype("float64"))
    else:
        inp_col = column_view_from_column(inp)

    if op == "count":
        min_periods = 0

    cdef gdf_index_type c_min_periods = min_periods

    if window == 0:
        data = rmm.device_array_like(inp.data.mem)
        if op in ["count", "sum"]:
            cudautils.fill_value(data, 0)
            mask = None
        else:
            cudautils.fill_value(data, inp.default_na_value())
            mask = cudautils.make_empty_mask(len(inp))
    else:
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

    result = Column.from_mem_views(data, mask)

    free(output_col)

    if op == "count":
        result = result.fillna(0)

    return result
