# Copyright (c) 2018, NVIDIA CORPORATION.

# cython: profile=False
# distutils: language = c++
# cython: embedsignature = True
# cython: language_level = 3

# Copyright (c) 2018, NVIDIA CORPORATION.

import numpy as np

from librmm_cffi import librmm as rmm
import nvcategory
import nvstrings

from cudf.bindings.cudf_cpp cimport *
from cudf.bindings.cudf_cpp import *
from cudf.bindings.join cimport *
from libcpp.vector cimport vector
from libc.stdint cimport uintptr_t
from libc.stdlib cimport calloc, malloc, free
cimport cython


@cython.boundscheck(False)
cpdef join(col_lhs, col_rhs, left_on, right_on, how, method):
    """
      Call gdf join for full outer, inner and left joins.
      Returns a list of tuples [(column, valid, name), ...]
    """

    # TODO: `context` leaks if exiting this function prematurely
    cdef gdf_context* context = create_context_view(0, method, 0, 0, 0,
                                                    'null_as_largest')

    if how not in ['left', 'inner', 'outer']:
        msg = "new join api only supports left, inner or outer"
        raise ValueError(msg)

    cdef vector[int] left_idx
    cdef vector[int] right_idx

    assert(len(left_on) == len(right_on))

    cdef vector[gdf_column*] list_lhs
    cdef vector[gdf_column*] list_rhs
    cdef vector[gdf_column*] result_cols

    result_col_names = []  # Preserve the order of the column names

    for name, col in col_lhs.items():
        check_gdf_compatibility(col)
        list_lhs.push_back(column_view_from_column(col._column))

        if name not in left_on:
            result_cols.push_back(
                column_view_from_NDArrays(
                    0,
                    None,
                    mask=None,
                    dtype=col._column.dtype,
                    null_count=0
                )
            )
            result_col_names.append(name)

    for name in left_on:
        result_cols.push_back(
            column_view_from_NDArrays(
                0,
                None,
                mask=None,
                dtype=col_lhs[name]._column.dtype,
                null_count=0
            )
        )
        result_col_names.append(name)
        left_idx.push_back(list(col_lhs.keys()).index(name))

    for name in right_on:
        right_idx.push_back(list(col_rhs.keys()).index(name))

    for name, col in col_rhs.items():
        check_gdf_compatibility(col)
        list_rhs.push_back(column_view_from_column(col._column))

        if name not in right_on:
            result_cols.push_back(
                column_view_from_NDArrays(
                    0,
                    None,
                    mask=None,
                    dtype=col._column.dtype,
                    null_count=0
                )
            )
            result_col_names.append(name)

    cdef gdf_error result = GDF_CUDA_ERROR
    cdef gdf_size_type col_lhs_len = len(col_lhs)
    cdef gdf_size_type col_rhs_len = len(col_rhs)
    cdef int c_num_cols_to_join = len(left_on)
    cdef int c_result_num_cols = result_cols.size()

    with nogil:
        if how == 'left':
            result = gdf_left_join(
                list_lhs.data(),
                col_lhs_len,
                left_idx.data(),
                list_rhs.data(),
                col_rhs_len,
                right_idx.data(),
                c_num_cols_to_join,
                c_result_num_cols,
                result_cols.data(),
                <gdf_column*> NULL,
                <gdf_column*> NULL,
                context
            )

        elif how == 'inner':
            result = gdf_inner_join(
                list_lhs.data(),
                col_lhs_len,
                left_idx.data(),
                list_rhs.data(),
                col_rhs_len,
                right_idx.data(),
                c_num_cols_to_join,
                c_result_num_cols,
                result_cols.data(),
                <gdf_column*> NULL,
                <gdf_column*> NULL,
                context
            )

        elif how == 'outer':
            result = gdf_full_join(
                list_lhs.data(),
                col_lhs_len,
                left_idx.data(),
                list_rhs.data(),
                col_rhs_len,
                right_idx.data(),
                c_num_cols_to_join,
                c_result_num_cols,
                result_cols.data(),
                <gdf_column*> NULL,
                <gdf_column*> NULL,
                context
            )

    check_gdf_error(result)

    res = []
    valids = []

    cdef uintptr_t data_ptr
    cdef uintptr_t valid_ptr

    for idx in range(result_cols.size()):
        col_dtype = np_dtype_from_gdf_column(result_cols[idx])
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
                # We need to create this just to make sure the memory is
                # properly freed
                tmp_data = rmm.device_array_from_ptr(
                    ptr=data_ptr,
                    nelem=result_cols[idx].size,
                    dtype='int32',
                    finalizer=rmm._make_finalizer(data_ptr, 0)
                )
            valid_ptr = <uintptr_t>result_cols[idx].valid
            if valid_ptr:
                valids.append(
                    rmm.device_array_from_ptr(
                        ptr=valid_ptr,
                        nelem=calc_chunk_size(
                            result_cols[idx].size,
                            mask_bitsize
                        ),
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
                        nelem=result_cols[idx].size,
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
                        nelem=calc_chunk_size(
                            result_cols[idx].size,
                            mask_bitsize
                        ),
                        dtype=mask_dtype,
                        finalizer=rmm._make_finalizer(valid_ptr, 0)
                    )
                )
            else:
                valids.append(None)

    free(context)
    for c_col in list_lhs:
        free_column(c_col)
    for c_col in list_rhs:
        free_column(c_col)
    for c_col in result_cols:
        free_column(c_col)

    return list(zip(res, valids, result_col_names))
