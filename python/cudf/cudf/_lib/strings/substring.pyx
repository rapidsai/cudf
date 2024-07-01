# Copyright (c) 2020-2024, NVIDIA CORPORATION.

import numpy as np

from cudf.core.buffer import acquire_spill_lock

from cudf._lib.column cimport Column

from cudf._lib.scalar import as_device_scalar

from cudf._lib.scalar cimport DeviceScalar

import cudf._lib.pylibcudf as plc


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
    cdef DeviceScalar start_scalar = as_device_scalar(start, np.int32)
    cdef DeviceScalar end_scalar = as_device_scalar(end, np.int32)
    cdef DeviceScalar step_scalar = as_device_scalar(step, np.int32)

    return Column.from_pylibcudf(
        plc.strings.slice.slice_strings(
            source_strings.to_pylibcudf(mode="read"),
            start_scalar.c_value,
            end_scalar.c_value,
            step_scalar.c_value
        )
    )


@acquire_spill_lock()
def slice_from(Column source_strings,
               Column starts,
               Column stops):
    """
    Returns a Column by extracting a substring of each string
    at given starts and stops positions. `starts` and `stops`
    here are positions per element in the string-column.
    """
    return Column.from_pylibcudf(
        plc.strings.slice.slice_strings(
            source_strings.to_pylibcudf(mode="read"),
            starts.to_pylibcudf(mode="read"),
            stops.to_pylibcudf(mode="read")
        )
    )


@acquire_spill_lock()
def get(Column source_strings,
        object index):
    """
    Returns a Column which contains only single
    character from each input string. The index of
    characters required can be controlled by passing `index`.
    """

    if index < 0:
        next_index = index - 1
        step = -1
    else:
        next_index = index + 1
        step = 1
    cdef DeviceScalar start_scalar = as_device_scalar(index, np.int32)
    cdef DeviceScalar end_scalar = as_device_scalar(next_index, np.int32)
    cdef DeviceScalar step_scalar = as_device_scalar(step, np.int32)

    return Column.from_pylibcudf(
        plc.strings.slice.slice_strings(
            source_strings.to_pylibcudf(mode="read"),
            start_scalar.c_value,
            end_scalar.c_value,
            step_scalar.c_value
        )
    )
