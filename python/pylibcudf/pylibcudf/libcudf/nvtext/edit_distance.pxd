# Copyright (c) 2020-2024, NVIDIA CORPORATION.
from libcpp cimport bool
from libcpp.memory cimport unique_ptr
from pylibcudf.exception_handler cimport libcudf_exception_handler
from pylibcudf.libcudf.column.column cimport column
from pylibcudf.libcudf.column.column_view cimport column_view


cdef extern from "nvtext/edit_distance.hpp" namespace "nvtext" nogil:

    cdef unique_ptr[column] edit_distance(
        const column_view & strings,
        const column_view & targets
    ) except +libcudf_exception_handler

    cdef unique_ptr[column] edit_distance_matrix(
        const column_view & strings
    ) except +libcudf_exception_handler
