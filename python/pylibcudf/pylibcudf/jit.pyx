# Copyright (c) 2025, NVIDIA CORPORATION.

from pylibcudf.libcudf cimport jit
from libcpp cimport bool

__all__ = ["is_runtime_jit_supported"]


cpdef bool is_runtime_jit_supported():
    with nogil:
        return jit.is_runtime_jit_supported()
