# Copyright (c) 2020-2024, NVIDIA CORPORATION.

from cudf.core.buffer import acquire_spill_lock

from libcpp.memory cimport unique_ptr
from libcpp.utility cimport move

cimport cudf._lib.cpp.filling as cpp_filling
from cudf._lib.column cimport Column
from cudf._lib.cpp.column.column cimport column
from cudf._lib.cpp.column.column_view cimport column_view
from cudf._lib.cpp.scalar.scalar cimport scalar
from cudf._lib.cpp.table.table cimport table
from cudf._lib.cpp.table.table_view cimport table_view
from cudf._lib.cpp.types cimport size_type
from cudf._lib.scalar cimport DeviceScalar
from cudf._lib.utils cimport columns_from_unique_ptr, table_view_from_columns

from cudf._lib import pylibcudf
from cudf._lib.scalar import as_device_scalar


@acquire_spill_lock()
def fill_in_place(Column destination, int begin, int end, DeviceScalar value):
    pylibcudf.filling.fill_in_place(
        destination.to_pylibcudf(mode='write'),
        begin,
        end,
        (<DeviceScalar> as_device_scalar(value, dtype=destination.dtype)).c_value
    )


@acquire_spill_lock()
def fill(Column destination, int begin, int end, DeviceScalar value):
    return Column.from_pylibcudf(
        pylibcudf.filling.fill(
            destination.to_pylibcudf(mode='read'),
            begin,
            end,
            (<DeviceScalar> as_device_scalar(value)).c_value
        )
    )


@acquire_spill_lock()
def repeat(list inp, object count):
    if isinstance(count, Column):
        return _repeat_via_column(inp, count)
    else:
        return _repeat_via_size_type(inp, count)


def _repeat_via_column(list inp, Column count):
    cdef table_view c_inp = table_view_from_columns(inp)
    cdef column_view c_count = count.view()
    cdef unique_ptr[table] c_result

    with nogil:
        c_result = move(cpp_filling.repeat(
            c_inp,
            c_count,
        ))

    return columns_from_unique_ptr(move(c_result))


def _repeat_via_size_type(list inp, size_type count):
    cdef table_view c_inp = table_view_from_columns(inp)
    cdef unique_ptr[table] c_result

    with nogil:
        c_result = move(cpp_filling.repeat(
            c_inp,
            count
        ))

    return columns_from_unique_ptr(move(c_result))


@acquire_spill_lock()
def sequence(int size, DeviceScalar init, DeviceScalar step):
    cdef size_type c_size = size
    cdef const scalar* c_init = init.get_raw_ptr()
    cdef const scalar* c_step = step.get_raw_ptr()
    cdef unique_ptr[column] c_result

    with nogil:
        c_result = move(cpp_filling.sequence(
            c_size,
            c_init[0],
            c_step[0]
        ))

    return Column.from_unique_ptr(move(c_result))
