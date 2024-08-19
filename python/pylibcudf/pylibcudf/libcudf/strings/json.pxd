# Copyright (c) 2021-2024, NVIDIA CORPORATION.

from libcpp cimport bool
from libcpp.memory cimport unique_ptr
from libcpp.string cimport string
from pylibcudf.libcudf.column.column cimport column
from pylibcudf.libcudf.column.column_view cimport column_view
from pylibcudf.libcudf.scalar.scalar cimport scalar, string_scalar


cdef extern from "cudf/json/json.hpp" namespace "cudf" nogil:
    cdef cppclass get_json_object_options:
        get_json_object_options() except +
        # getters
        bool get_allow_single_quotes() except +
        bool get_strip_quotes_from_single_strings() except +
        bool get_missing_fields_as_nulls() except +
        # setters
        void set_allow_single_quotes(bool val) except +
        void set_strip_quotes_from_single_strings(bool val) except +
        void set_missing_fields_as_nulls(bool val) except +

    cdef unique_ptr[column] get_json_object(
        column_view col,
        string_scalar json_path,
        get_json_object_options options,
    ) except +
