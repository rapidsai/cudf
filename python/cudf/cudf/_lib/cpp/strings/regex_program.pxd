# Copyright (c) 2022, NVIDIA CORPORATION.

from libcpp.memory cimport unique_ptr
from libcpp.string cimport string

from cudf._lib.cpp.strings.regex_flags cimport regex_flags


cdef extern from "cudf/strings/regex/regex_program.hpp" \
        namespace "cudf::strings" nogil:

    cdef cppclass regex_program:

        @staticmethod
        unique_ptr[regex_program] create(string pattern, regex_flags flags)
