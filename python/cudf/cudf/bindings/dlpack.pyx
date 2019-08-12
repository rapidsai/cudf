# Copyright (c) 2018, NVIDIA CORPORATION.

# cython: profile=False
# distutils: language = c++
# cython: embedsignature = True
# cython: language_level = 3

# Copyright (c) 2018, NVIDIA CORPORATION.

from cudf.bindings.cudf_cpp cimport *
from cudf.bindings.cudf_cpp import *
from cudf.bindings.dlpack cimport *
from librmm_cffi import librmm as rmm

from libc.stdint cimport uintptr_t, int8_t
from libc.stdlib cimport calloc, malloc, free
from cpython cimport pycapsule

import numpy as np
import pandas as pd
import pyarrow as pa
import warnings


cpdef from_dlpack(dlpack_capsule):
    """
    Converts a DLPack Tensor PyCapsule into a list of cudf Column objects.

    DLPack Tensor PyCapsule is expected to have the name "dltensor".
    """
    warnings.warn("WARNING: cuDF from_dlpack() assumes column-major (Fortran"
                  " order) input. If the input tensor is row-major, transpose"
                  " it before passing it to this function.")

    cdef DLManagedTensor* dlpack_tensor = <DLManagedTensor*>pycapsule.\
        PyCapsule_GetPointer(dlpack_capsule, 'dltensor')
    pycapsule.PyCapsule_SetName(dlpack_capsule, 'used_dltensor')
    cdef gdf_size_type c_result_num_cols
    cdef gdf_column* result_cols

    with nogil:
        result = gdf_from_dlpack(
            &result_cols,
            &c_result_num_cols,
            dlpack_tensor
        )

    check_gdf_error(result)

    # TODO: Replace this with a generic function from cudf_cpp.pyx since this
    # is copied from join.pyx
    res = []
    valids = []

    for idx in range(c_result_num_cols):
        data_ptr = <uintptr_t>result_cols[idx].data
        if data_ptr:
            res.append(
                rmm.device_array_from_ptr(
                    ptr=data_ptr,
                    nelem=result_cols[idx].size,
                    dtype=np_dtype_from_gdf_column(&result_cols[idx]),
                    finalizer=rmm._make_finalizer(data_ptr, 0)
                )
            )
        else:
            res.append(
                rmm.device_array(
                    0,
                    dtype=np_dtype_from_gdf_column(&result_cols[idx])
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


cpdef to_dlpack(in_cols):
    """
    Converts a a list of cudf Column objects into a DLPack Tensor PyCapsule.

    DLPack Tensor PyCapsule will have the name "dltensor".
    """

    warnings.warn("WARNING: cuDF to_dlpack() produces column-major (Fortran "
                  "order) output. If the output tensor needs to be row major, "
                  "transpose the output of this function.")

    input_num_cols = len(in_cols)

    cdef DLManagedTensor* dlpack_tensor =<DLManagedTensor*>malloc(
        sizeof(DLManagedTensor)
    )
    cdef gdf_column** input_cols = <gdf_column**>malloc(
        input_num_cols * sizeof(gdf_column*)
    )
    cdef gdf_size_type c_input_num_cols = input_num_cols

    for idx, col in enumerate(in_cols):
        check_gdf_compatibility(col)
        input_cols[idx] = column_view_from_column(col)

    with nogil:
        result = gdf_to_dlpack(
            dlpack_tensor,
            input_cols,
            c_input_num_cols
        )

    check_gdf_error(result)

    return pycapsule.PyCapsule_New(
        dlpack_tensor,
        'dltensor',
        dlmanaged_tensor_pycapsule_deleter
    )

cdef void dlmanaged_tensor_pycapsule_deleter(object pycap_obj):
    cdef DLManagedTensor* dlpack_tensor= <DLManagedTensor*>0
    try:
        dlpack_tensor = <DLManagedTensor*>pycapsule.PyCapsule_GetPointer(
            pycap_obj, 'used_dltensor')
        return  # we do not call a used capsule's deleter
    except Exception:
        dlpack_tensor = <DLManagedTensor*>pycapsule.PyCapsule_GetPointer(
            pycap_obj, 'dltensor')
    dlpack_tensor.deleter(dlpack_tensor)
