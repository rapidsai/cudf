# Copyright (c) 2022-2025, NVIDIA CORPORATION.
from libc.stdint cimport int32_t
from pylibcudf.exception_handler cimport libcudf_exception_handler


cdef extern from "cudf/strings/regex/flags.hpp" namespace "cudf::strings" nogil:

    # Unlike most enums in pylibcudf, this one needs to be declared not as a C++-style
    # scoped enum (enum class) because it is used as a bitmask and therefore
    # needs to support bitwise operations producing values outside the defined set.
    # Cython will generate IntFlag enums for "enum", whereas it will generate IntEnum
    # (which does not support this use case) for "enum class".
    # https://github.com/cython/cython/pull/4877#issuecomment-1213227726
    cpdef enum regex_flags:
        DEFAULT "cudf::strings::regex_flags::DEFAULT"
        MULTILINE "cudf::strings::regex_flags::MULTILINE"
        DOTALL "cudf::strings::regex_flags::DOTALL"
