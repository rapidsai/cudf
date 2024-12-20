# Copyright (c) 2021-2024, NVIDIA CORPORATION.
from cudf.core.buffer import acquire_spill_lock

from pylibcudf.libcudf.types cimport size_type

from cudf._lib.column cimport Column

import pylibcudf as plc


@acquire_spill_lock()
def repeat_scalar(Column source_strings,
                  size_type repeats):
    """
    Returns a Column after repeating
    each string in `source_strings`
    `repeats` number of times.
    """
    plc_result = plc.strings.repeat.repeat_strings(
        source_strings.to_pylibcudf(mode="read"),
        repeats
    )
    return Column.from_pylibcudf(plc_result)


@acquire_spill_lock()
def repeat_sequence(Column source_strings,
                    Column repeats):
    """
    Returns a Column after repeating
    each string in `source_strings`
    `repeats` number of times.
    """
    plc_result = plc.strings.repeat.repeat_strings(
        source_strings.to_pylibcudf(mode="read"),
        repeats.to_pylibcudf(mode="read")
    )
    return Column.from_pylibcudf(plc_result)
