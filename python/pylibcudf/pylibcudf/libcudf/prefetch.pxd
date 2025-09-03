# Copyright (c) 2022-2025, NVIDIA CORPORATION.


cdef extern from "cudf/utilities/prefetch.hpp" namespace "cudf::prefetch" nogil:
    void enable() noexcept
    void disable() noexcept
    void enable_debugging() noexcept
    void disable_debugging() noexcept
