from libc.stdlib cimport calloc, malloc, free
from libc.stdint cimport uintptr_t

import numba.cuda

from cudf.dataframe.column import Column

from cudf.bindings.cudf_cpp cimport *
from cudf.bindings.cudf_cpp import *
from cudf.bindings.rolling cimport *


def apply_rolling(inp, window, min_periods, center, op):
    cdef gdf_column *inp_col
    cdef gdf_column *output_col = <gdf_column*> malloc(sizeof(gdf_column*))
    cdef gdf_index_type c_window = 0
    cdef gdf_index_type c_forward_window = 0
    cdef gdf_agg_op c_op = agg_ops[op]
    cdef gdf_index_type *c_window_col = NULL
    cdef gdf_index_type *c_min_periods_col = NULL
    cdef gdf_index_type *c_forward_window_col = NULL

    if op == "mean":
        inp_col = column_view_from_column(inp.astype("float64"))
    else:
        inp_col = column_view_from_column(inp)

    if op == "count":
        min_periods = 0

    cdef gdf_index_type c_min_periods = min_periods

    cdef uintptr_t c_window_ptr
    if isinstance(window, numba.cuda.devicearray.DeviceNDArray):
        if center:
            # TODO: we can support this even though Pandas currently does not
            raise NotImplementedError(
                "center is not implemented for offset-based windows"
            )
        c_window_ptr = get_ctype_ptr(window)
        c_window_col = <gdf_index_type*> c_window_ptr
    else:
        if center:
            c_window = (window // 2) + 1
            c_forward_window = window - (c_window)
        else:
            c_window = window
            c_forward_window = 0

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
                                        c_forward_window,
                                        c_op,
                                        c_window_col,
                                        c_min_periods_col,
                                        c_forward_window_col
            )
        data, mask = gdf_column_to_column_mem(output_col)

    result = Column.from_mem_views(data, mask)

    if c_window_col is NULL:
        # Pandas only does this for fixed windows...?
        if op == "count":
            result = result.fillna(0)

    free(output_col)
    free(inp_col)

    return result
