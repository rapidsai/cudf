# Copyright (c) 2020-2024, NVIDIA CORPORATION.

from cudf.core.buffer import acquire_spill_lock


from cudf._lib.column cimport Column
from cudf._lib.scalar cimport DeviceScalar
from cudf._lib.utils cimport columns_from_pylibcudf_table

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
    ctbl = pylibcudf.Table([col.to_pylibcudf(mode="read") for col in inp])
    if isinstance(count, Column):
        count = count.to_pylibcudf(mode="read")
    return columns_from_pylibcudf_table(
        pylibcudf.filling.repeat(
            ctbl,
            count
        )
    )


@acquire_spill_lock()
def sequence(int size, DeviceScalar init, DeviceScalar step):
    return Column.from_pylibcudf(
        pylibcudf.filling.sequence(
            size,
            (<DeviceScalar> as_device_scalar(init)).c_value,
            (<DeviceScalar> as_device_scalar(step)).c_value
        )
    )
