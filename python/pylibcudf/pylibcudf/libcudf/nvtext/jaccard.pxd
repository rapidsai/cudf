# Copyright (c) 2023-2024, NVIDIA CORPORATION.
from libcpp.memory cimport unique_ptr
from pylibcudf.exception_handler cimport libcudf_exception_handler
from pylibcudf.libcudf.column.column cimport column
from pylibcudf.libcudf.column.column_view cimport column_view
from pylibcudf.libcudf.types cimport size_type


cdef extern from "nvtext/jaccard.hpp" namespace "nvtext" nogil:

    cdef unique_ptr[column] jaccard_index(
        const column_view &input1,
        const column_view &input2,
        size_type width
    ) except +libcudf_exception_handler
