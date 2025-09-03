# Copyright (c) 2022-2025, NVIDIA CORPORATION.


cdef extern from "cudf/utilities/prefetch.hpp" namespace "cudf::prefetch" nogil:
    void enable()
    void disable()
    void enable_debugging()
    void disable_debugging()
