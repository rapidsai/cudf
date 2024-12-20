# Copyright (c) 2020-2024, NVIDIA CORPORATION.

from libc.stdint cimport uint32_t

from cudf.core.buffer import acquire_spill_lock

from cudf._lib.column cimport Column

from pylibcudf.strings import contains
from pylibcudf.strings.regex_program import RegexProgram


@acquire_spill_lock()
def contains_re(Column source_strings, object reg_ex, uint32_t flags):
    """
    Returns a Column of boolean values with True for `source_strings`
    that contain regular expression `reg_ex`.
    """
    prog = RegexProgram.create(str(reg_ex), flags)
    return Column.from_pylibcudf(
        contains.contains_re(source_strings.to_pylibcudf(mode="read"), prog)
    )


@acquire_spill_lock()
def count_re(Column source_strings, object reg_ex, uint32_t flags):
    """
    Returns a Column with count of occurrences of `reg_ex` in
    each string of `source_strings`
    """
    prog = RegexProgram.create(str(reg_ex), flags)
    return Column.from_pylibcudf(
        contains.count_re(source_strings.to_pylibcudf(mode="read"), prog)
    )


@acquire_spill_lock()
def match_re(Column source_strings, object reg_ex, uint32_t flags):
    """
    Returns a Column with each value True if the string matches `reg_ex`
    regular expression with each record of `source_strings`
    """
    prog = RegexProgram.create(str(reg_ex), flags)
    return Column.from_pylibcudf(
        contains.matches_re(source_strings.to_pylibcudf(mode="read"), prog)
    )


@acquire_spill_lock()
def like(Column source_strings, object py_pattern, object py_escape):
    """
    Returns a Column with each value True if the string matches the
    `py_pattern` like expression with each record of `source_strings`
    """
    plc_column = contains.like(
        source_strings.to_pylibcudf(mode="read"),
        py_pattern.device_value.c_value,
        py_escape.device_value.c_value,
    )
    return Column.from_pylibcudf(plc_column)
