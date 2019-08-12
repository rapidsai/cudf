# Copyright (c) 2019, NVIDIA CORPORATION.

# cython: profile=False
# distutils: language = c++
# cython: embedsignature = True
# cython: language_level = 3

from libc.stdint cimport uintptr_t

import numba.cuda

from cudf.dataframe.column import Column

from cudf.bindings.cudf_cpp cimport *
from cudf.bindings.cudf_cpp import *
from cudf.bindings.rolling cimport *


def apply_rolling(inp, window, min_periods, center, op):
    cdef gdf_column *c_input_col
    cdef gdf_column* c_output_col = NULL
    cdef gdf_index_type c_window = 0
    cdef gdf_index_type c_forward_window = 0
    cdef gdf_agg_op c_op = agg_ops[op]
    cdef gdf_index_type *c_window_col = NULL
    cdef gdf_index_type *c_min_periods_col = NULL
    cdef gdf_index_type *c_forward_window_col = NULL

    if op == "mean":
        inp = inp.astype("float64")

    c_input_col = column_view_from_column(inp, inp.name)

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

    result = None

    if window == 0:
        mask = None
        out_value = 0
        null_count = 0
        out_size = inp.data.mem.size
        out_dtype = inp.data.mem.dtype
        if op not in ["count", "sum"]:
            null_count = len(inp)
            out_value = inp.default_na_value()
            mask = cudautils.make_empty_mask(null_count)
        data = cudautils.full(out_size, out_value, out_dtype)
        result = Column.from_mem_views(data, mask, null_count, inp.name)
    else:
        with nogil:
            c_output_col = rolling_window(
                c_input_col[0],
                c_window,
                c_min_periods,
                c_forward_window,
                c_op,
                c_window_col,
                c_min_periods_col,
                c_forward_window_col
            )
        # I'd expect this to work but it doesn't...
        # result = gdf_column_to_column(c_output_col)
        data, mask = gdf_column_to_column_mem(c_output_col)
        result = Column.from_mem_views(data, mask, None, inp.name)

    if c_window_col is NULL and op == "count":
        # Pandas only does this for fixed windows...?
        result = result.fillna(0)

    free_column(c_input_col)
    free_column(c_output_col)

    return result
