# Copyright (c) 2020-2024, NVIDIA CORPORATION.

from libcpp.memory cimport unique_ptr
from libcpp.utility cimport move

from cudf.core.buffer import acquire_spill_lock

from cudf._lib.column cimport Column
from cudf._lib.pylibcudf.libcudf.column.column_view cimport column_view
from cudf._lib.pylibcudf.libcudf.scalar.scalar cimport string_scalar
from cudf._lib.pylibcudf.libcudf.strings.split.partition cimport (
    partition as cpp_partition,
    rpartition as cpp_rpartition,
)
from cudf._lib.pylibcudf.libcudf.table.table cimport table
from cudf._lib.scalar cimport DeviceScalar
from cudf._lib.utils cimport data_from_unique_ptr


@acquire_spill_lock()
def partition(Column source_strings,
              object py_delimiter):
    """
    Returns data by splitting the `source_strings`
    column at the first occurrence of the specified `py_delimiter`.
    """

    cdef DeviceScalar delimiter = py_delimiter.device_value

    cdef unique_ptr[table] c_result
    cdef column_view source_view = source_strings.view()
    cdef const string_scalar* scalar_str = <const string_scalar*>(
        delimiter.get_raw_ptr()
    )

    with nogil:
        c_result = move(cpp_partition(
            source_view,
            scalar_str[0]
        ))

    return data_from_unique_ptr(
        move(c_result),
        column_names=range(0, c_result.get()[0].num_columns())
    )


@acquire_spill_lock()
def rpartition(Column source_strings,
               object py_delimiter):
    """
    Returns a Column by splitting the `source_strings`
    column at the last occurrence of the specified `py_delimiter`.
    """

    cdef DeviceScalar delimiter = py_delimiter.device_value

    cdef unique_ptr[table] c_result
    cdef column_view source_view = source_strings.view()
    cdef const string_scalar* scalar_str = <const string_scalar*>(
        delimiter.get_raw_ptr()
    )

    with nogil:
        c_result = move(cpp_rpartition(
            source_view,
            scalar_str[0]
        ))

    return data_from_unique_ptr(
        move(c_result),
        column_names=range(0, c_result.get()[0].num_columns())
    )
