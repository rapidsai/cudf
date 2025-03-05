# Copyright (c) 2025, NVIDIA CORPORATION.
from libcpp.memory cimport unique_ptr
from pylibcudf.exception_handler cimport libcudf_exception_handler
from pylibcudf.libcudf.column.column cimport column
from pylibcudf.libcudf.column.column_view cimport column_view
from pylibcudf.libcudf.types cimport size_type


cdef extern from "nvtext/wordpiece_tokenize.hpp" namespace "nvtext" nogil:

    cdef struct wordpiece_vocabulary "nvtext::wordpiece_vocabulary":
        pass

    cdef unique_ptr[wordpiece_vocabulary] load_wordpiece_vocabulary(
        const column_view & strings
    ) except +libcudf_exception_handler

    cdef unique_ptr[column] wordpiece_tokenize(
        const column_view & strings,
        const wordpiece_vocabulary & vocabulary,
        size_type max_tokens_per_row
    ) except +libcudf_exception_handler
