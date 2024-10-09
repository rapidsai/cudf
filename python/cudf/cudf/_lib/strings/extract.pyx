# Copyright (c) 2020-2024, NVIDIA CORPORATION.

from libc.stdint cimport uint32_t

from cudf.core.buffer import acquire_spill_lock

from cudf._lib.column cimport Column

import pylibcudf as plc


@acquire_spill_lock()
def extract(Column source_strings, object pattern, uint32_t flags):
    """
    Returns data which contains extracted capture groups provided in
    `pattern` for all `source_strings`.
    The returning data contains one row for each subject string,
    and one column for each group.
    """
    prog = plc.strings.regex_program.RegexProgram.create(str(pattern), flags)
    plc_result = plc.strings.extract.extract(
        source_strings.to_pylibcudf(mode="read"), prog
    )
    return dict(enumerate(Column.from_pylibcudf(col) for col in plc_result.columns()))
