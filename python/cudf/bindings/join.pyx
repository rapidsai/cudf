# Copyright (c) 2018, NVIDIA CORPORATION.

# cython: profile=False
# distutils: language = c++
# cython: embedsignature = True
# cython: language_level = 3

# Copyright (c) 2018, NVIDIA CORPORATION.

from cudf.bindings.cudf_cpp cimport *
from cudf.bindings.cudf_cpp import *

import numpy as np
import pandas as pd
import pyarrow as pa

from librmm_cffi import librmm as rmm
import nvcategory
import nvstrings

from libc.stdint cimport uintptr_t
from libc.stdlib cimport calloc, malloc, free

cimport cython


@cython.boundscheck(False)
cpdef join(col_lhs, col_rhs, left_on, right_on, how, method='sort'):
    """
      Call gdf join for full outer, inner and left joins.
    """

    cdef gdf_context* context = create_context_view(0, method, 0, 0, 0)

    if how not in ['left', 'inner', 'outer']:
        msg = "new join api only supports left, inner or outer"
        raise ValueError(msg)

    result_col_names = []

    cdef int[::1] left_idx = np.zeros(len(left_on), dtype=np.dtype("int32"))
    cdef int[::1] right_idx = np.zeros(len(right_on), dtype=np.dtype("int32"))

    on = list(set(left_on + right_on))
    num_cols_to_join = len(on)
    result_num_cols = len(col_lhs) + len(col_rhs) - num_cols_to_join

    cdef gdf_column** list_lhs = <gdf_column**>malloc(len(col_lhs) * sizeof(gdf_column*))
    cdef gdf_column** list_rhs = <gdf_column**>malloc(len(col_rhs) * sizeof(gdf_column*))
    cdef gdf_column** result_cols = <gdf_column**>malloc(result_num_cols * sizeof(gdf_column*))

    cdef int res_idx = 0
    cdef int idx = 0

    for name, col in col_lhs.items():
        check_gdf_compatibility(col)
        list_lhs[idx] = column_view_from_column(col._column)

        mask_size = 0
        if col.has_null_mask:
            mask_size = col.nullmask.size

        if name not in left_on:
            result_cols[res_idx] = column_view_from_NDArrays(0, None, mask=None, dtype=col._column.dtype, null_count=0)
            result_col_names.append(name)
            res_idx = res_idx + 1
        idx = idx + 1

    idx = 0
    for name in list(set(left_on + right_on)):
        # TODO: Need careful type promotion here between lhs and rhs
        if name in left_on:
            dtype = col_lhs[name]._column.dtype
        else:
            dtype = col_rhs[name]._column.dtype
        result_cols[res_idx] = column_view_from_NDArrays(0, None, mask=None, dtype=dtype, null_count=0)
        result_col_names.append(name)
        left_idx[idx] = list(col_lhs.keys()).index(name)
        right_idx[idx] = list(col_rhs.keys()).index(name)
        res_idx = res_idx + 1
        idx = idx + 1

    idx = 0
    for name, col in col_rhs.items():
        check_gdf_compatibility(col)
        list_rhs[idx] = column_view_from_column(col._column)

        mask_size = 0
        if col.has_null_mask:
            mask_size = col.nullmask.size

        if name not in right_on:
            result_cols[res_idx] = column_view_from_NDArrays(0, None, mask=None, dtype=col._column.dtype, null_count=0)
            result_col_names.append(name)
            res_idx = res_idx + 1
        idx = idx + 1

    cdef gdf_error result = GDF_CUDA_ERROR
    cdef gdf_size_type col_lhs_len = len(col_lhs)
    cdef gdf_size_type col_rhs_len = len(col_rhs)
    cdef int c_num_cols_to_join = num_cols_to_join
    cdef int c_result_num_cols = result_num_cols

    with nogil:
        if how == 'left':
            result = gdf_left_join(list_lhs,
                col_lhs_len,
                &left_idx[0],
                list_rhs,
                col_rhs_len,
                &right_idx[0],
                c_num_cols_to_join,
                c_result_num_cols,
                result_cols,
                <gdf_column*> NULL,
                <gdf_column*> NULL,
                context)

        elif how == 'inner':
            result = gdf_inner_join(list_lhs,
                col_lhs_len,
                &left_idx[0],
                list_rhs,
                col_rhs_len,
                &right_idx[0],
                c_num_cols_to_join,
                c_result_num_cols,
                result_cols,
                <gdf_column*> NULL,
                <gdf_column*> NULL,
                context)

        elif how == 'outer':
            result = gdf_full_join(list_lhs,
                col_lhs_len,
                &left_idx[0],
                list_rhs,
                col_rhs_len,
                &right_idx[0],
                c_num_cols_to_join,
                c_result_num_cols,
                result_cols,
                <gdf_column*> NULL,
                <gdf_column*> NULL,
                context)

    check_gdf_error(result)

    res = []
    valids = []

    cdef uintptr_t data_ptr
    cdef uintptr_t valid_ptr

    for idx in range(result_num_cols):
        col_dtype = gdf_to_np_dtype(result_cols[idx].dtype)
        if col_dtype == np.object_:
            nvcat_ptr = <uintptr_t> result_cols[idx].dtype_info.category
            if nvcat_ptr:
                nvcat_obj = nvcategory.bind_cpointer(int(nvcat_ptr))
                nvstr_obj = nvcat_obj.to_strings()
            else:
                nvstr_obj = nvstrings.to_device([])
            res.append(nvstr_obj)
            data_ptr = <uintptr_t>result_cols[idx].data
            if data_ptr:
            # We need to create this just to make sure the memory is properly freed
                tmp_data = rmm.device_array_from_ptr(
                    ptr=data_ptr,
                    nelem= result_cols[idx].size,
                    dtype='int32',
                    finalizer=rmm._make_finalizer(data_ptr, 0)
                )
            valid_ptr = <uintptr_t>result_cols[idx].valid
            if valid_ptr:
                valids.append(
                    rmm.device_array_from_ptr(
                        ptr=valid_ptr,
                        nelem=calc_chunk_size(result_cols[idx].size, mask_bitsize),
                        dtype=mask_dtype,
                        finalizer=rmm._make_finalizer(valid_ptr, 0)
                    )
                )
            else:
                valids.append(None)
        else:
            data_ptr = <uintptr_t>result_cols[idx].data
            if data_ptr:
                res.append(
                    rmm.device_array_from_ptr(
                        ptr=data_ptr,
                        nelem= result_cols[idx].size,
                        dtype=col_dtype,
                        finalizer=rmm._make_finalizer(data_ptr, 0)
                    )
                )
            else:
                res.append(
                    rmm.device_array(
                        0,
                        dtype=col_dtype
                    )
                )
            valid_ptr = <uintptr_t>result_cols[idx].valid
            if valid_ptr:
                valids.append(
                    rmm.device_array_from_ptr(
                        ptr=valid_ptr,
                        nelem=calc_chunk_size(result_cols[idx].size, mask_bitsize),
                        dtype=mask_dtype,
                        finalizer=rmm._make_finalizer(valid_ptr, 0)
                    )
                )
            else:
                valids.append(None)

    return res, valids
