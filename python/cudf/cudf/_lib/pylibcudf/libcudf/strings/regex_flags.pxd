# Copyright (c) 2022-2024, NVIDIA CORPORATION.

from libc.stdint cimport int32_t


cdef extern from "cudf/strings/regex/flags.hpp" \
        namespace "cudf::strings" nogil:

    cpdef enum class regex_flags(int32_t):
        DEFAULT
        MULTILINE
        DOTALL
