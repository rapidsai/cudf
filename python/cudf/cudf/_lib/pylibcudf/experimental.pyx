# Copyright (c) 2024, NVIDIA CORPORATION.

from libcpp cimport bool
from libcpp.string cimport string

from cudf._lib.pylibcudf.libcudf cimport experimental as cpp_experimental


cpdef enable_prefetching(str key):
    cdef string c_key = key.encode("utf-8")
    cpp_experimental.enable_prefetching(c_key)

cpdef disable_prefetching(str key):
    cdef string c_key = key.encode("utf-8")
    cpp_experimental.disable_prefetching(c_key)

cpdef prefetch_debugging(bool enable):
    cpp_experimental.prefetch_debugging(enable)
