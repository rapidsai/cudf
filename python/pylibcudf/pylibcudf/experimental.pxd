# Copyright (c) 2024, NVIDIA CORPORATION.

from libcpp cimport bool


cpdef enable_prefetching(str key)

cpdef disable_prefetching(str key)

cpdef prefetch_debugging(bool enable)
