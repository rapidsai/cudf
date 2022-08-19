# Copyright (c) 2021-2022, NVIDIA CORPORATION.
#
# distutils: language = c++
# cython: c_string_type=unicode, c_string_encoding=utf8

import os

from libcpp.memory cimport unique_ptr
from libcpp.string cimport string
from libcpp.utility cimport move
from libcpp.vector cimport vector

from cudf.core.buffer import Buffer

from rmm._lib.device_buffer cimport DeviceBuffer, device_buffer
from strings_udf._lib.cpp.strings_udf cimport (
    to_string_view_array as cpp_to_string_view_array,
)

from cudf._lib.column cimport Column
from cudf._lib.cpp.column.column cimport column
from cudf._lib.cpp.column.column_view cimport column_view
from cudf._lib.cpp.types cimport size_type

import numpy as np


def to_string_view_array(Column strings_col):
    cdef unique_ptr[device_buffer] c_buffer

    # with nogil:
    c_buffer = move(cpp_to_string_view_array(strings_col.view()))

    buffer = DeviceBuffer.c_from_unique_ptr(move(c_buffer))
    buffer = Buffer(buffer)
    return buffer
