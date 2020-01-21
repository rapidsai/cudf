# Copyright (c) 2019, NVIDIA CORPORATION.

# cython: profile=False
# distutils: language = c++
# cython: embedsignature = True
# cython: language_level = 3

from libc.stdint cimport uintptr_t
from libcpp.string cimport string

import numba.cuda
import numba.numpy_support

from cudf.utils import cudautils

from cudf._lib.cudf cimport *
from cudf._lib.cudf import *
cimport cudf._lib.includes.rolling as cpp_rolling


def rolling(inp, window, min_periods, center, op):
    from cudf.core.buffer import Buffer
    from cudf.core.column import build_column

    cdef gdf_column* c_out_ptr = NULL
    cdef size_type c_window = 0
    cdef size_type c_forward_window = 0
    cdef gdf_agg_op c_op
    cdef size_type *c_window_col = NULL
    cdef size_type *c_min_periods_col = NULL
    cdef size_type *c_forward_window_col = NULL

    cdef string cpp_str
    cdef gdf_dtype g_type

    if op == "mean":
        inp = inp.astype("float64")

    cdef gdf_column* c_in_col = column_view_from_column(inp)

    if op == "count":
        min_periods = 0

    cdef size_type c_min_periods = min_periods

    cdef uintptr_t c_window_ptr
    if isinstance(window, numba.cuda.devicearray.DeviceNDArray):
        if center:
            # TODO: we can support this even though Pandas currently does not
            raise NotImplementedError(
                "center is not implemented for offset-based windows"
            )
        c_window_ptr = get_ctype_ptr(window)
        c_window_col = <size_type*> c_window_ptr
    else:
        if center:
            c_window = (window // 2) + 1
            c_forward_window = window - (c_window)
        else:
            c_window = window
            c_forward_window = 0

    data = None
    mask = None
    out_col = None
    null_count = None

    if window == 0:
        fill_value = 0
        null_count = 0
        if op not in ["count", "sum"]:
            null_count = len(inp)
            fill_value = inp.default_na_value()
            mask = cudautils.make_empty_mask(null_count)
        data = cudautils.full(
            inp.data_array_view.size, fill_value, inp.data_array_view.dtype
        )

        out_col = build_column(Buffer(data),
                               dtype=data.dtype,
                               mask=Buffer(mask))
    else:
        if callable(op):
            nb_type = numba.numpy_support.from_dtype(inp.dtype)
            type_signature = (nb_type[:],)
            compiled_op = cudautils.compile_udf(op, type_signature)
            cpp_str = compiled_op[0].encode('UTF-8')
            if compiled_op[1] not in dtypes:
                raise TypeError(
                    "Result of window function has unsupported dtype {}"
                    .format(op[1])
                )
            g_type = dtypes[compiled_op[1]]
            with nogil:
                c_out_col = cpp_rolling.rolling_window(
                    c_in_col[0],
                    c_window,
                    c_min_periods,
                    c_forward_window,
                    cpp_str,
                    GDF_NUMBA_GENERIC_AGG_OPS,
                    g_type,
                    c_window_col,
                    c_min_periods_col,
                    c_forward_window_col
                )
            out_col = gdf_column_to_column(&c_out_col)
        else:
            c_op = agg_ops[op]
            with nogil:
                c_out_ptr = cpp_rolling.rolling_window(
                    c_in_col[0],
                    c_window,
                    c_min_periods,
                    c_forward_window,
                    c_op,
                    c_window_col,
                    c_min_periods_col,
                    c_forward_window_col
                )
            out_col = gdf_column_to_column(c_out_ptr)

    if c_window_col is NULL and op == "count":
        # Pandas only does this for fixed windows...?
        out_col = out_col.fillna(0)

    free_column(c_in_col)
    free_column(c_out_ptr)

    return out_col
