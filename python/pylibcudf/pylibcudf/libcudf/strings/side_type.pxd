# Copyright (c) 2022-2024, NVIDIA CORPORATION.
from pylibcudf.exception_handler import libcudf_exception_handler
from libcpp cimport int


cdef extern from "cudf/strings/side_type.hpp" namespace "cudf::strings" nogil:

    cpdef enum class side_type(int):
        LEFT
        RIGHT
        BOTH
