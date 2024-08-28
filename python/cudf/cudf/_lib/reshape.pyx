# Copyright (c) 2019-2024, NVIDIA CORPORATION.

from cudf.core.buffer import acquire_spill_lock

from pylibcudf.libcudf.types cimport size_type

from cudf._lib.column cimport Column
from cudf._lib.utils cimport columns_from_pylibcudf_table

import pylibcudf as plc


@acquire_spill_lock()
def interleave_columns(list source_columns):
    return Column.from_pylibcudf(
        plc.reshape.interleave_columns(
            plc.Table([
                c.to_pylibcudf(mode="read") for c in source_columns
            ])
        )
    )


@acquire_spill_lock()
def tile(list source_columns, size_type count):
    cdef size_type c_count = count

    return columns_from_pylibcudf_table(
        plc.reshape.tile(
            plc.Table([
                c.to_pylibcudf(mode="read") for c in source_columns
            ]),
            c_count
        )
    )
