# Copyright (c) 2021-2024, NVIDIA CORPORATION.

from cudf.core.buffer import acquire_spill_lock

from cudf._lib.column cimport Column

import pylibcudf as plc


@acquire_spill_lock()
def is_float(Column source_strings):
    """
    Returns a Column of boolean values with True for `source_strings`
    that have floats.
    """
    plc_column = plc.strings.convert.convert_floats.is_float(
        source_strings.to_pylibcudf(mode="read")
    )
    return Column.from_pylibcudf(plc_column)
