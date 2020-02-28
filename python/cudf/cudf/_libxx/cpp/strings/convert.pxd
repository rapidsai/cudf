# Copyright (c) 2020, NVIDIA CORPORATION.

from cudf._libxx.cpp.column.column cimport column
from cudf._libxx.cpp.column.column_view cimport column_view
from cudf._libxx.cpp.scalar.scalar cimport string_scalar
from cudf._libxx.cpp.types cimport data_type

from libcpp.memory cimport unique_ptr
from libcpp.string cimport string

cdef extern from "cudf/strings/convert/convert_booleans.hpp" namespace \
        "cudf::strings" nogil:
    cdef unique_ptr[column] to_booleans(
        column_view input_col,
        string_scalar true_string)

    cdef unique_ptr[column] from_booleans(
        column_view input_col,
        string_scalar true_string,
        string_scalar false_string)

cdef extern from "cudf/strings/convert/convert_datetime.hpp" namespace \
        "cudf::strings" nogil:
    cdef unique_ptr[column] to_timestamps(
        column_view input_col,
        data_type timestamp_type,
        string format)

    cdef unique_ptr[column] from_timestamps(
        column_view input_col,
        string format)

cdef extern from "cudf/strings/convert/convert_floats.hpp" namespace \
        "cudf::strings" nogil:
    cdef unique_ptr[column] to_floats(
        column_view input_col,
        data_type output_type)

    cdef unique_ptr[column] from_floats(
        column_view input_col)

cdef extern from "cudf/strings/convert/convert_integers.hpp" namespace \
        "cudf::strings" nogil:
    cdef unique_ptr[column] to_integers(
        column_view input_col,
        data_type output_type)

    cdef unique_ptr[column] from_integers(
        column_view input_col)

    cdef unique_ptr[column] hex_to_integers(
        column_view input_col,
        data_type output_type)

cdef extern from "cudf/strings/convert/convert_ipv4.hpp" namespace \
        "cudf::strings" nogil:
    cdef unique_ptr[column] ipv4_to_integers(
        column_view input_col)

    cdef unique_ptr[column] integers_to_ipv4(
        column_view input_col)

cdef extern from "cudf/strings/convert/convert_urls.hpp" namespace \
        "cudf::strings" nogil:
    cdef unique_ptr[column] url_encode(
        column_view input_col)

    cdef unique_ptr[column] url_decode(
        column_view input_col)
