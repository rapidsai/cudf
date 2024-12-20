# Copyright (c) 2021-2024, NVIDIA CORPORATION.

from cudf.core.buffer import acquire_spill_lock

import pylibcudf as plc

from cudf._lib.column cimport Column


@acquire_spill_lock()
def is_integer(Column source_strings):
    """
    Returns a Column of boolean values with True for `source_strings`
    that have integers.
    """
    return Column.from_pylibcudf(
        plc.strings.convert.convert_integers.is_integer(
            source_strings.to_pylibcudf(mode="read")
        )
    )
