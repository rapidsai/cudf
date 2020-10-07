# Copyright (c) 2020, NVIDIA CORPORATION.

from cudf._lib.cpp.column.column_view cimport column_view
from libcpp.memory cimport unique_ptr
from libcpp.utility cimport move
from cudf._lib.column cimport Column
from cudf._lib.cpp.types cimport size_type
from cudf._lib.cpp.column.column cimport column
import numpy as np

from cudf._lib.cpp.strings.substring cimport (
    slice_strings as cpp_slice_strings
)

from cudf._lib.scalar import as_scalar
from cudf._lib.scalar cimport Scalar
from cudf._lib.cpp.scalar.scalar cimport numeric_scalar


def slice_strings(Column source_strings,
                  object start,
                  object end,
                  object step):
    """
    Returns a Column by extracting a substring of each string
    at given start and end positions. Slicing can also be
    performed in steps by skipping `step` number of
    characters in a string.
    """
    cdef unique_ptr[column] c_result
    cdef column_view source_view = source_strings.view()

    cdef Scalar start_scalar = as_scalar(start, np.int32)
    cdef Scalar end_scalar = as_scalar(end, np.int32)
    cdef Scalar step_scalar = as_scalar(step, np.int32)

    cdef numeric_scalar[size_type]* start_numeric_scalar = \
        <numeric_scalar[size_type]*>(start_scalar.c_value.get())
    cdef numeric_scalar[size_type]* end_numeric_scalar = \
        <numeric_scalar[size_type]*>(end_scalar.c_value.get())
    cdef numeric_scalar[size_type]* step_numeric_scalar = \
        <numeric_scalar[size_type]*>(step_scalar.c_value.get())

    with nogil:
        c_result = move(cpp_slice_strings(
            source_view,
            start_numeric_scalar[0],
            end_numeric_scalar[0],
            step_numeric_scalar[0]
        ))

    return Column.from_unique_ptr(move(c_result))


def slice_from(Column source_strings,
               Column starts,
               Column stops):
    """
    Returns a Column by extracting a substring of each string
    at given starts and stops positions. `starts` and `stops`
    here are positions per element in the string-column.
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


def get(Column source_strings,
        object index):
    """
    Returns a Column which contains only single
    charater from each input string. The index of
    characters required can be controlled by passing `index`.
    """
    cdef unique_ptr[column] c_result
    cdef column_view source_view = source_strings.view()
    if index < 0:
        next_index = index - 1
        step = -1
    else:
        next_index = index + 1
        step = 1
    cdef Scalar start_scalar = as_scalar(index, np.int32)
    cdef Scalar end_scalar = as_scalar(next_index, np.int32)
    cdef Scalar step_scalar = as_scalar(step, np.int32)

    cdef numeric_scalar[size_type]* start_numeric_scalar = \
        <numeric_scalar[size_type]*>(start_scalar.c_value.get())
    cdef numeric_scalar[size_type]* end_numeric_scalar = \
        <numeric_scalar[size_type]*>(end_scalar.c_value.get())
    cdef numeric_scalar[size_type]* step_numeric_scalar = \
        <numeric_scalar[size_type]*>(step_scalar.c_value.get())

    with nogil:
        c_result = move(cpp_slice_strings(
            source_view,
            start_numeric_scalar[0],
            end_numeric_scalar[0],
            step_numeric_scalar[0]
        ))

    return Column.from_unique_ptr(move(c_result))
