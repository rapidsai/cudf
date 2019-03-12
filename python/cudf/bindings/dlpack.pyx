# Copyright (c) 2018, NVIDIA CORPORATION.

# cython: profile=False
# distutils: language = c++
# cython: embedsignature = True
# cython: language_level = 3

# Copyright (c) 2018, NVIDIA CORPORATION.

from cudf.bindings.cudf_cpp cimport *
from cudf.bindings.cudf_cpp import *
from cudf.bindings.dlpack cimport DLManagedTensor
from librmm_cffi import librmm as rmm

from libc.stdint cimport uintptr_t, int8_t
from libc.stdlib cimport calloc, malloc, free
from cpython cimport pycapsule

import numpy as np
import pandas as pd
import pyarrow as pa


cpdef from_dlpack(dlpack_capsule):
    """
    Converts a DLPack Tensor PyCapsule into a list of cudf Column objects.
    """
    cdef DLManagedTensor* dlpack_tensor = <DLManagedTensor*>pycapsule.PyCapsule_GetPointer(dlpack_capsule, 'dltensor')
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
                    dtype=gdf_to_np_dtype( result_cols[idx].dtype),
                    finalizer=rmm._make_finalizer(data_ptr, 0)
                )
            )
        else:
            res.append(
                rmm.device_array(
                    0,
                    dtype=gdf_to_np_dtype( result_cols[idx].dtype)
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
            valids.append(
                rmm.device_array(
                    0,
                    dtype=mask_dtype
                )
            )

    return res, valids


cpdef to_dlpack(in_cols):

    input_num_cols = len(in_cols)

    cdef DLManagedTensor* dlpack_tensor = <DLManagedTensor*>malloc(sizeof(DLManagedTensor))
    cdef gdf_column** input_cols = <gdf_column**>malloc(input_num_cols * sizeof(gdf_column*))
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

    print("cudf to_dlpack")

    return pycapsule.PyCapsule_New(dlpack_tensor, 'dltensor', pycapsule_deleter)

cdef void pycapsule_deleter(object pycap_obj):
    print("cudf to_dlpack pycapsule_deleter")
    cdef DLManagedTensor* dlpack_tensor= <DLManagedTensor*>0
    try:
        dlpack_tensor = <DLManagedTensor*>pycapsule.PyCapsule_GetPointer(
            pycap_obj, 'used_dltensor')
        return # we do not call a used capsule's deleter
    except Exception:
        dlpack_tensor = <DLManagedTensor*>pycapsule.PyCapsule_GetPointer(
            pycap_obj, 'dltensor')
    dlpack_tensor.deleter(dlpack_tensor)


###
### Temporary Testing Code
###


cdef class DLPackMemory:

    """Memory object for a dlpack tensor.
    This does not allocate any memory.
    """

    cdef DLManagedTensor* dlm_tensor
    cdef object dltensor
    cdef int device_id
    cdef size_t ptr
    cdef size_t size

    def __init__(self, object dltensor):
        self.dltensor = dltensor
        print("start")
        self.dlm_tensor = <DLManagedTensor *>pycapsule.PyCapsule_GetPointer(
            dltensor, 'dltensor')
        print("1")
        self.device_id = self.dlm_tensor.dl_tensor.ctx.device_id
        print("2")
        self.ptr = <size_t>self.dlm_tensor.dl_tensor.data
        print("3")
        cdef int n = 0
        print("4 ", n)
        cdef int ndim = self.dlm_tensor.dl_tensor.ndim
        print("5 ", ndim)
        cdef int64_t* shape = self.dlm_tensor.dl_tensor.shape
        print("6 ", shape[0])
        print("7 ", self.dlm_tensor.dl_tensor.dtype.bits)
        for s in shape[:ndim]:
            n += s
        self.size = self.dlm_tensor.dl_tensor.dtype.bits * n // 8
        print("8 ", self.size)
        # Make sure this capsule will never be used again.
        pycapsule.PyCapsule_SetName(dltensor, 'used_dltensor')
        print("9 ", "used")

    def __dealloc__(self):
        print("dealloc")
        self.dlm_tensor.deleter(self.dlm_tensor)

cpdef int test_fromDlpack(object dltensor):
    # dlm_tensor = <DLManagedTensor *>cpython.PyCapsule_GetPointer(
    #         dltensor, 'dltensor')
    mem = DLPackMemory(dltensor)

    cdef DLDataType dtype = mem.dlm_tensor.dl_tensor.dtype
    cdef int bits = dtype.bits
    if dtype.code == DLDataTypeCode.kDLUInt:
        if bits == 8:
            cp_dtype = cupy.uint8
        elif bits == 16:
            cp_dtype = cupy.uint16
        elif bits == 32:
            cp_dtype = cupy.uint32
        elif bits == 64:
            cp_dtype = cupy.uint64
        else:
            raise TypeError('uint{} is not supported.'.format(bits))
    elif dtype.code == DLDataTypeCode.kDLInt:
        if bits == 8:
            cp_dtype = cupy.int8
        elif bits == 16:
            cp_dtype = cupy.int16
        elif bits == 32:
            cp_dtype = cupy.int32
        elif bits == 64:
            cp_dtype = cupy.int64
        else:
            raise TypeError('int{} is not supported.'.format(bits))
    elif dtype.code == DLDataTypeCode.kDLFloat:
        if bits == 16:
            cp_dtype = cupy.float16
        elif bits == 32:
            cp_dtype = cupy.float32
        elif bits == 64:
            cp_dtype = cupy.float64
        else:
            raise TypeError('float{} is not supported.'.format(bits))
    else:
        raise TypeError('Unsupported dtype. dtype code: {}'.format(dtype.code))

    mem_ptr = <ptrdiff_t>mem.dlm_tensor.dl_tensor.data + <ptrdiff_t>mem.dlm_tensor.dl_tensor.byte_offset
    cdef int64_t ndim = mem.dlm_tensor.dl_tensor.ndim

    cdef int64_t* shape = mem.dlm_tensor.dl_tensor.shape
    cdef vector[Py_ssize_t] shape_vec
    shape_vec.assign(shape, shape + ndim)

    cdef int64_t* strides = mem.dlm_tensor.dl_tensor.strides
    cdef vector[Py_ssize_t] strides_vec
    for i in range(ndim):
        strides_vec.push_back(strides[i] * (bits // 8))

    return 0