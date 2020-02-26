# Copyright (c) 2018-2020, NVIDIA CORPORATION.

from cudf._libxx.lib cimport *
from cudf._libxx.lib import *
from cudf._libxx.column cimport Column
from cudf._libxx.scalar cimport Scalar
from cudf._libxx.scalar import Scalar

from cudf._libxx.strings.substring cimport (
    slice_strings as cpp_slice_strings
)


def slice_strings(Column source_strings,
                  size_type start,
                  size_type end,
                  size_type step):
    """
    Returns a Column by extracting a substring of each string
    at given start and end positions. Slicing can also be
    performed in steps by skipping `step` number of
    characters in a string.
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
        size_type index):
    """
    Returns a Column which contains only single
    charater from each input string. The index of
    characters required can be controlled by passing `index`.
    """
    cdef unique_ptr[column] c_result
    cdef column_view source_view = source_strings.view()

    with nogil:
        c_result = move(cpp_slice_strings(
            source_view,
            index,
            index + 1,
            1
        ))

    return Column.from_unique_ptr(move(c_result))
