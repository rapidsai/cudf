# Copyright (c) 2020-2024, NVIDIA CORPORATION.

from cudf.core.buffer import acquire_spill_lock

from cudf._lib.column cimport Column

import pylibcudf as plc


@acquire_spill_lock()
def find_multiple(Column source_strings, Column target_strings):
    """
    Returns a column with character position values where each
    of the `target_strings` are found in each string of `source_strings`.
    """
    plc_result = plc.strings.find_multiple.find_multiple(
        source_strings.to_pylibcudf(mode="read"),
        target_strings.to_pylibcudf(mode="read")
    )
    return Column.from_pylibcudf(plc_result)
