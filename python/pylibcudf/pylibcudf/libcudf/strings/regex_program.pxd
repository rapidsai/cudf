# Copyright (c) 2022-2024, NVIDIA CORPORATION.
from libcpp.memory cimport unique_ptr
from libcpp.string cimport string
from pylibcudf.exception_handler cimport libcudf_exception_handler
from pylibcudf.libcudf.strings.regex_flags cimport regex_flags


cdef extern from "cudf/strings/regex/regex_program.hpp" \
        namespace "cudf::strings" nogil:

    cdef cppclass regex_program:

        @staticmethod
        unique_ptr[regex_program] create(
            string pattern,
            regex_flags flags
        ) except +libcudf_exception_handler
