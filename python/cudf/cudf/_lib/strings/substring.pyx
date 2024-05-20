# Copyright (c) 2020-2024, NVIDIA CORPORATION.

import numpy as np

from libcpp.memory cimport unique_ptr
from libcpp.utility cimport move

from cudf.core.buffer import acquire_spill_lock

from cudf._lib.column cimport Column
from cudf._lib.pylibcudf.libcudf.column.column cimport column
from cudf._lib.pylibcudf.libcudf.column.column_view cimport column_view
from cudf._lib.pylibcudf.libcudf.strings.substring cimport (
    slice_strings as cpp_slice_strings,
)
from cudf._lib.pylibcudf.libcudf.types cimport size_type

from cudf._lib.scalar import as_device_scalar

from cudf._lib.pylibcudf.libcudf.scalar.scalar cimport numeric_scalar
from cudf._lib.scalar cimport DeviceScalar


@acquire_spill_lock()
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

    cdef DeviceScalar start_scalar = as_device_scalar(start, np.int32)
    cdef DeviceScalar end_scalar = as_device_scalar(end, np.int32)
    cdef DeviceScalar step_scalar = as_device_scalar(step, np.int32)

    cdef numeric_scalar[size_type]* start_numeric_scalar = \
        <numeric_scalar[size_type]*>(
            start_scalar.get_raw_ptr())
    cdef numeric_scalar[size_type]* end_numeric_scalar = \
        <numeric_scalar[size_type]*>(end_scalar.get_raw_ptr())
    cdef numeric_scalar[size_type]* step_numeric_scalar = \
        <numeric_scalar[size_type]*>(step_scalar.get_raw_ptr())

    with nogil:
        c_result = move(cpp_slice_strings(
            source_view,
            start_numeric_scalar[0],
            end_numeric_scalar[0],
            step_numeric_scalar[0]
        ))

    return Column.from_unique_ptr(move(c_result))


@acquire_spill_lock()
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


@acquire_spill_lock()
def get(Column source_strings,
        object index):
    """
    Returns a Column which contains only single
    character from each input string. The index of
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
    cdef DeviceScalar start_scalar = as_device_scalar(index, np.int32)
    cdef DeviceScalar end_scalar = as_device_scalar(next_index, np.int32)
    cdef DeviceScalar step_scalar = as_device_scalar(step, np.int32)

    cdef numeric_scalar[size_type]* start_numeric_scalar = \
        <numeric_scalar[size_type]*>(
            start_scalar.get_raw_ptr())
    cdef numeric_scalar[size_type]* end_numeric_scalar = \
        <numeric_scalar[size_type]*>(end_scalar.get_raw_ptr())
    cdef numeric_scalar[size_type]* step_numeric_scalar = \
        <numeric_scalar[size_type]*>(step_scalar.get_raw_ptr())

    with nogil:
        c_result = move(cpp_slice_strings(
            source_view,
            start_numeric_scalar[0],
            end_numeric_scalar[0],
            step_numeric_scalar[0]
        ))

    return Column.from_unique_ptr(move(c_result))
