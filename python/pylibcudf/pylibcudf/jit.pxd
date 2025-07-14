# Copyright (c) 2025, NVIDIA CORPORATION.

from libcpp cimport bool
from libc.stdint cimport int8_t


cpdef enum class udf_source_type(int8_t):
    CUDA
    PTX

cpdef bool is_runtime_jit_supported()
