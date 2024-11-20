# Copyright (c) 2021-2024, NVIDIA CORPORATION.
from libcpp.memory cimport unique_ptr
from pylibcudf.exception_handler cimport libcudf_exception_handler
from pylibcudf.libcudf.column.column cimport column
from pylibcudf.libcudf.column.column_view cimport column_view


cdef extern from "cudf/lists/filling.hpp" namespace "cudf::lists" nogil:
    cdef unique_ptr[column] sequences(
        const column_view& starts,
        const column_view& sizes,
    ) except +libcudf_exception_handler

    cdef unique_ptr[column] sequences(
        const column_view& starts,
        const column_view& steps,
        const column_view& sizes,
    ) except +libcudf_exception_handler
