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

from cudf._lib.column cimport Column
from cudf._lib.cpp.column.column cimport column
from cudf._lib.cpp.column.column_view cimport column_view
from cudf._lib.cpp.types cimport size_type
from strings_udf._lib.cpp.strings_udf cimport (
    call_udf as cpp_call_udf,
    create_udf_module as cpp_create_udf_module,
    from_dstring_array as cpp_from_dstring_array,
    to_string_view_array as cpp_to_string_view_array,
    udf_module as cpp_udf_module,
)

import numpy as np


def process_udf(udf, name, cols):
    cdef string c_udf
    cdef string c_name
    cdef size_type c_size

    cdef vector[column_view] c_columns
    cdef unique_ptr[cpp_udf_module] c_module
    cdef vector[string] c_options

    cdef unique_ptr[column] c_result

    c_udf = udf.encode('UTF-8')
    c_name = name.encode('UTF-8')

    cdef Column col = cols[0]._column
    c_size = col.size
    for c in cols:
        col = c._column
        c_columns.push_back(col.view())

    include_path = "-I" + os.environ.get("CONDA_PREFIX") + "/include"
    c_options.push_back(str(include_path).encode('UTF-8'))

    # with nogil:
    c_module = move(cpp_create_udf_module(c_udf, c_options))
    # c_module will be nullptr if there is a compile error

    # with nogil:
    c_result = move(cpp_call_udf(c_module.get()[0], c_name, c_size, c_columns))

    return Column.from_unique_ptr(move(c_result))


def to_string_view_array(Column strings_col):
    cdef unique_ptr[device_buffer] c_buffer

    # with nogil:
    c_buffer = move(cpp_to_string_view_array(strings_col.view()))

    buffer = DeviceBuffer.c_from_unique_ptr(move(c_buffer))
    buffer = Buffer(buffer)
    return buffer


def from_dstring_array(DeviceBuffer d_buffer):
    cdef size_t size = d_buffer.c_size()
    cdef void* data = d_buffer.c_data()
    cdef unique_ptr[column] c_result
    # data = <void *>

    # with nogil:
    c_result = move(cpp_from_dstring_array(data, size))

    return Column.from_unique_ptr(move(c_result))
