# Copyright (c) 2019-2024, NVIDIA CORPORATION.

from libc.stdint cimport uint32_t

from cudf.core.buffer import acquire_spill_lock

from cudf._lib.column cimport Column

import pylibcudf as plc


@acquire_spill_lock()
def findall(Column source_strings, object pattern, uint32_t flags):
    """
    Returns data with all non-overlapping matches of `pattern`
    in each string of `source_strings` as a lists column.
    """
    prog = plc.strings.regex_program.RegexProgram.create(
        str(pattern), flags
    )
    plc_result = plc.strings.findall.findall(
        source_strings.to_pylibcudf(mode="read"),
        prog,
    )
    return Column.from_pylibcudf(plc_result)


@acquire_spill_lock()
def find_re(Column source_strings, object pattern, uint32_t flags):
    """
    Returns character positions where the pattern first matches
    the elements in source_strings.
    """
    prog = plc.strings.regex_program.RegexProgram.create(
        str(pattern), flags
    )
    plc_result = plc.strings.findall.find_re(
        source_strings.to_pylibcudf(mode="read"),
        prog,
    )
    return Column.from_pylibcudf(plc_result)
