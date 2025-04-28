# Copyright (c) 2025, NVIDIA CORPORATION.

from libcpp.memory cimport unique_ptr
from libcpp.pair cimport pair
from pylibcudf.exception_handler cimport libcudf_exception_handler
from pylibcudf.libcudf.column.column cimport column
from pylibcudf.libcudf.column.column_view cimport column_view
from pylibcudf.libcudf.types cimport size_type

from rmm.librmm.device_uvector cimport device_uvector

ctypedef unique_ptr[device_uvector[size_type]] suffix_array_type

cdef extern from "nvtext/dedup.hpp" namespace "nvtext" nogil:

    cdef unique_ptr[column] substring_duplicates(
        column_view source_strings,
        size_type min_width) except +libcudf_exception_handler

    cdef suffix_array_type build_suffix_array(
        column_view source_strings,
        size_type min_width) except +libcudf_exception_handler
