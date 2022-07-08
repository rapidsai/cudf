# Copyright (c) 2021, NVIDIA CORPORATION.

from libcpp.memory cimport unique_ptr
from libcpp.string cimport string

from cudf._lib.cpp.column.column cimport column
from cudf._lib.cpp.column.column_view cimport column_view
from cudf._lib.cpp.scalar.scalar cimport scalar, string_scalar


cdef extern from "cudf/strings/json.hpp" namespace "cudf::strings" nogil:
    cdef unique_ptr[column] get_json_object(
        column_view col,
        string_scalar json_path,
    ) except +
