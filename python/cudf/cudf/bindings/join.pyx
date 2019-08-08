# Copyright (c) 2018, NVIDIA CORPORATION.

# cython: profile=False
# distutils: language = c++
# cython: embedsignature = True
# cython: language_level = 3

# Copyright (c) 2018, NVIDIA CORPORATION.

from __future__ import print_function
import numpy as np

from librmm_cffi import librmm as rmm
import nvcategory
import nvstrings

from cudf.bindings.cudf_cpp cimport *
from cudf.bindings.cudf_cpp import *
from cudf.bindings.join cimport *
from cudf.bindings.utils cimport *
from libcpp.utility cimport pair
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
    cdef vector[int] left_common_name_join_idx
    cdef vector[int] right_common_name_join_idx

    assert(len(left_on) == len(right_on))

    cdef cudf_table *list_lhs = table_from_dataframe(col_lhs)
    cdef cudf_table *list_rhs = table_from_dataframe(col_rhs)

    result_col_names = []  # Preserve the order of the column names

    for name, col in col_lhs.items():
        check_gdf_compatibility(col)
        result_col_names.append(name)

    for name in left_on:
        # This will ensure that the column name is valid
        col_lhs[name]
        left_idx.push_back(list(col_lhs.keys()).index(name))
        if (name in right_on and
           (left_on.index(name) == right_on.index(name))):
            left_common_name_join_idx.push_back(
                list(col_lhs.keys()).index(name))

    for name in right_on:
        # This will ensure that the column name is valid
        col_rhs[name]
        right_idx.push_back(list(col_rhs.keys()).index(name))
        if (name in left_on and (left_on.index(name)
           == right_on.index(name))):
            right_common_name_join_idx.push_back(
                list(col_rhs.keys()).index(name))

    for name, col in col_rhs.items():
        check_gdf_compatibility(col)
        if not ((name in left_on) and (name in right_on)
           and (left_on.index(name) == right_on.index(name))):
            result_col_names.append(name)

    cdef pair[cudf_table, cudf_table] result

    with nogil:
        if how == 'left':
            result = gdf_left_join(
                list_lhs[0],
                left_idx,
                list_rhs[0],
                right_idx,
                <gdf_column*> NULL,
                <gdf_column*> NULL,
                left_common_name_join_idx,
                right_common_name_join_idx,
                context
            )

        elif how == 'inner':
            result = gdf_inner_join(
                list_lhs[0],
                left_idx,
                list_rhs[0],
                right_idx,
                <gdf_column*> NULL,
                <gdf_column*> NULL,
                left_common_name_join_idx,
                right_common_name_join_idx,
                context
            )

        elif how == 'outer':
            result = gdf_full_join(
                list_lhs[0],
                left_idx,
                list_rhs[0],
                right_idx,
                <gdf_column*> NULL,
                <gdf_column*> NULL,
                left_common_name_join_idx,
                right_common_name_join_idx,
                context
            )

    res = []
    valids = []
    cdef vector[gdf_column*] result_cols

    for idx in range(result.first.num_columns()):
        result_cols.push_back(result.first.get_column(idx))

    for idx in range(result.second.num_columns()):
        result_cols.push_back(result.second.get_column(idx))

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

    del list_lhs
    del list_rhs

    return list(zip(res, valids, result_col_names))
