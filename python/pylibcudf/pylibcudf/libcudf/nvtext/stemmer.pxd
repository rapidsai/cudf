# Copyright (c) 2020-2024, NVIDIA CORPORATION.

from libc.stdint cimport int32_t
from libcpp cimport bool
from libcpp.memory cimport unique_ptr
from pylibcudf.libcudf.column.column cimport column
from pylibcudf.libcudf.column.column_view cimport column_view
from pylibcudf.libcudf.types cimport size_type


cdef extern from "nvtext/stemmer.hpp" namespace "nvtext" nogil:
    cpdef enum class letter_type:
        CONSONANT
        VOWEL

    cdef unique_ptr[column] porter_stemmer_measure(
        const column_view & strings
    ) except +

    cdef unique_ptr[column] is_letter(
        column_view source_strings,
        letter_type ltype,
        size_type character_index) except +

    cdef unique_ptr[column] is_letter(
        column_view source_strings,
        letter_type ltype,
        column_view indices) except +

ctypedef int32_t underlying_type_t_letter_type
