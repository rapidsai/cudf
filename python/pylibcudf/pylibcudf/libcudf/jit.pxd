# Copyright (c) 2025, NVIDIA CORPORATION.

from libcpp cimport bool
from libc.stdint cimport int32_t
from pylibcudf.exception_handler cimport libcudf_exception_handler

cdef extern from "cudf/jit/runtime_support.hpp" namespace "cudf" nogil:

    cdef bool is_runtime_jit_supported() except +libcudf_exception_handler



cdef extern from "cudf/jit/udf.hpp" namespace "cudf" nogil:

    cpdef enum class udf_source_type(int32_t):
        CUDA
        PTX
