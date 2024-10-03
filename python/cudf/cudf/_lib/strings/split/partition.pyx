# Copyright (c) 2020-2024, NVIDIA CORPORATION.

from cudf.core.buffer import acquire_spill_lock

from cudf._lib.column cimport Column

import pylibcudf as plc


@acquire_spill_lock()
def partition(Column source_strings,
              object py_delimiter):
    """
    Returns data by splitting the `source_strings`
    column at the first occurrence of the specified `py_delimiter`.
    """
    plc_table = plc.strings.split.partition.partition(
        source_strings.to_pylibcudf(mode="read"),
        py_delimiter.device_value.c_value
    )
    return dict(enumerate(Column.from_pylibcudf(col) for col in plc_table.columns()))


@acquire_spill_lock()
def rpartition(Column source_strings,
               object py_delimiter):
    """
    Returns a Column by splitting the `source_strings`
    column at the last occurrence of the specified `py_delimiter`.
    """
    plc_table = plc.strings.split.partition.rpartition(
        source_strings.to_pylibcudf(mode="read"),
        py_delimiter.device_value.c_value
    )
    return dict(enumerate(Column.from_pylibcudf(col) for col in plc_table.columns()))
