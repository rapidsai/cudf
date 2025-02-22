# Copyright (c) 2020-2025, NVIDIA CORPORATION.
from libcpp cimport bool
from libcpp.memory cimport unique_ptr
from pylibcudf.exception_handler cimport libcudf_exception_handler
from pylibcudf.libcudf.column.column cimport column
from pylibcudf.libcudf.column.column_view cimport column_view


cdef extern from "nvtext/normalize.hpp" namespace "nvtext" nogil:

    cdef unique_ptr[column] normalize_spaces(
        const column_view & strings
    ) except +libcudf_exception_handler

    cdef unique_ptr[column] normalize_characters(
        const column_view & strings,
        bool do_lower_case
    ) except +libcudf_exception_handler

    cdef struct character_normalizer "nvtext::character_normalizer":
        pass

    cdef unique_ptr[character_normalizer] create_character_normalizer(
        bool do_lower_case,
        const column_view & strings
    ) except +libcudf_exception_handler

    cdef unique_ptr[column] normalize_characters(
        const column_view & strings,
        const character_normalizer & normalizer
    ) except +libcudf_exception_handler
