# Copyright (c) 2019, NVIDIA CORPORATION.

# cython: profile=False
# distutils: language = c++
# cython: embedsignature = True
# cython: language_level = 3

from libc cimport stdlib
from libc.stdint cimport uint8_t
from libc.stdint cimport uint16_t
from libc.stdint cimport int64_t
from libc.stdint cimport uint64_t
from libcpp.vector cimport vector

cdef extern from "dlpack.h" nogil:
    
    cdef enum DLDeviceType:
        kDLCPU = 1
        kDLGPU = 2
        kDLCPUPinned = 3
        kDLOpenCL = 4
        kDLMetal = 8
        kDLVPI = 9
        kDLROCM = 10

    ctypedef struct DLContext:
        DLDeviceType device_type
        int device_id

    cdef enum DLDataTypeCode:
        kDLInt = <unsigned int>0
        kDLUInt = <unsigned int>1
        kDLFloat = <unsigned int>2

    ctypedef struct DLDataType:
        uint8_t code
        uint8_t bits
        uint16_t lanes

    ctypedef struct DLTensor:
        size_t data  # Safer than "void *"
        DLContext ctx
        int ndim
        DLDataType dtype
        int64_t* shape
        int64_t* strides
        uint64_t byte_offset

    ctypedef struct DLManagedTensor:
        DLTensor dl_tensor
        void* manager_ctx
        void(*deleter)(DLManagedTensor*)
