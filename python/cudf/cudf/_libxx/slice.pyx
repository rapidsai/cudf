# Copyright (c) 2018-2020, NVIDIA CORPORATION.

# cython: profile=False
# distutils: language = c++
# cython: embedsignature = True
# cython: language_level = 3

# Copyright (c) 2018, NVIDIA CORPORATION.
import cudf
from cudf._libxx.table cimport *
from cudf._libxx.lib cimport *
from cudf._libxx.lib import *
from cudf._libxx.scalar cimport Scalar
from cudf._libxx.scalar import Scalar

from cpython cimport pycapsule

import warnings

from cudf._libxx.strings.slice cimport (
    slice_strings as cpp_slice_strings,
    replace_slice as cpp_replace_slice
)


def slice_strings(Column source_strings,
                  size_type start,
                  size_type end,
                  size_type step):
    """
    Converts a DLPack Tensor PyCapsule into a cudf Table object.

    DLPack Tensor PyCapsule is expected to have the name "dltensor".
    """
    cdef unique_ptr[column] c_result
    cdef column_view source_view = source_strings.view()

    with nogil:
        c_result = move(cpp_slice_strings(
            source_view,
            start,
            end,
            step
        ))

    return Column.from_unique_ptr(move(c_result))


def slice_from(Column source_strings,
               Column starts,
               Column stops):
    """
    Converts a DLPack Tensor PyCapsule into a cudf Table object.

    DLPack Tensor PyCapsule is expected to have the name "dltensor".
    """
    cdef unique_ptr[column] c_result
    cdef column_view source_view = source_strings.view()
    cdef column_view starts_view = starts.view()
    cdef column_view stops_view = stops.view()

    with nogil:
        c_result = move(cpp_slice_strings(
            source_view,
            starts_view,
            stops_view
        ))

    return Column.from_unique_ptr(move(c_result))


def slice_replace(Column source_strings,
                  size_type start,
                  size_type stop,
                  Scalar repl):

    cdef unique_ptr[column] c_result
    cdef column_view source_view = source_strings.view()

    cdef string_scalar* x = <string_scalar*>(repl.c_value.get())

    with nogil:
        c_result = move(cpp_replace_slice(
            source_view,
            x[0],
            start,
            stop
        ))

    return Column.from_unique_ptr(move(c_result))


def insert(Column source_strings,
           size_type start,
           Scalar repl):

    cdef unique_ptr[column] c_result
    cdef column_view source_view = source_strings.view()

    cdef string_scalar* x = <string_scalar*>(repl.c_value.get())

    with nogil:
        c_result = move(cpp_replace_slice(
            source_view,
            x[0],
            start,
            start
        ))

    return Column.from_unique_ptr(move(c_result))


def get(Column source_strings,
        size_type i):
    """
    Converts a DLPack Tensor PyCapsule into a cudf Table object.

    DLPack Tensor PyCapsule is expected to have the name "dltensor".
    """
    cdef unique_ptr[column] c_result
    cdef column_view source_view = source_strings.view()

    with nogil:
        c_result = move(cpp_slice_strings(
            source_view,
            i,
            i+1,
            1
        ))

    return Column.from_unique_ptr(move(c_result))
