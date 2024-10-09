# Copyright (c) 2018-2024, NVIDIA CORPORATION.

from cudf.core.buffer import acquire_spill_lock

from cudf._lib.column cimport Column

import pylibcudf as plc


@acquire_spill_lock()
def count_characters(Column source_strings):
    """
    Returns an integer numeric column containing the
    length of each string in characters.
    """
    plc_column = plc.strings.attributes.count_characters(
        source_strings.to_pylibcudf(mode="read")
    )
    return Column.from_pylibcudf(plc_column)


@acquire_spill_lock()
def count_bytes(Column source_strings):
    """
    Returns an integer numeric column containing the
    number of bytes of each string.
    """
    plc_column = plc.strings.attributes.count_bytes(
        source_strings.to_pylibcudf(mode="read")
    )
    return Column.from_pylibcudf(plc_column)


@acquire_spill_lock()
def code_points(Column source_strings):
    """
    Creates a numeric column with code point values (integers)
    for each character of each string.
    """
    plc_column = plc.strings.attributes.code_points(
        source_strings.to_pylibcudf(mode="read")
    )
    return Column.from_pylibcudf(plc_column)
