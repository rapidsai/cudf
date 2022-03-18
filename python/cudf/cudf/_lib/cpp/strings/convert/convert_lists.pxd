# Copyright (c) 2021, NVIDIA CORPORATION.
from libcpp.memory cimport unique_ptr

from cudf._lib.cpp.column.column cimport column
from cudf._lib.cpp.column.column_view cimport column_view
from cudf._lib.cpp.scalar.scalar cimport string_scalar


cdef extern from "cudf/strings/convert/convert_lists.hpp" namespace \
        "cudf::strings" nogil:

    cdef unique_ptr[column] format_list_column(
        column_view input_col,
        string_scalar na_rep,
        column_view separators) except +
