# Copyright (c) 2024, NVIDIA CORPORATION.

from libcpp cimport bool
from libcpp.string cimport string

from cudf._lib.pylibcudf.libcudf cimport experimental as cpp_experimental


cpdef enable_prefetching(str key):
    # helper to convert a gather map to a Column
    cdef string c_key = key.encode("utf-8")
    cpp_experimental.enable_prefetching(c_key)

cpdef prefetch_debugging(bool enable):
    # helper to convert a gather map to a Column
    cpp_experimental.prefetch_debugging(enable)
