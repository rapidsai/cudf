# Copyright (c) 2020-2024, NVIDIA CORPORATION.
from cudf.core.buffer import acquire_spill_lock

from pylibcudf.libcudf.types cimport size_type

from cudf._lib.column cimport Column

import pylibcudf as plc


@acquire_spill_lock()
def pad(Column source_strings,
        size_type width,
        fill_char,
        side=plc.strings.side_type.SideType.LEFT):
    """
    Returns a Column by padding strings in `source_strings`
    up to the given `width`. Direction of padding is to be specified by `side`.
    The additional characters being filled can be changed by specifying
    `fill_char`.
    """
    plc_result = plc.strings.padding.pad(
        source_strings.to_pylibcudf(mode="read"),
        width,
        side,
        fill_char,
    )
    return Column.from_pylibcudf(plc_result)


@acquire_spill_lock()
def zfill(Column source_strings,
          size_type width):
    """
    Returns a Column by prepending strings in `source_strings`
    with '0' characters up to the given `width`.
    """
    plc_result = plc.strings.padding.zfill(
        source_strings.to_pylibcudf(mode="read"),
        width
    )
    return Column.from_pylibcudf(plc_result)


def center(Column source_strings,
           size_type width,
           fill_char):
    """
    Returns a Column by filling left and right side of strings
    in `source_strings` with additional character, `fill_char`
    up to the given `width`.
    """
    return pad(source_strings, width, fill_char, plc.strings.side_type.SideType.BOTH)


def ljust(Column source_strings,
          size_type width,
          fill_char):
    """
    Returns a Column by filling right side of strings in `source_strings`
    with additional character, `fill_char` up to the given `width`.
    """
    return pad(source_strings, width, fill_char, plc.strings.side_type.SideType.RIGHT)


def rjust(Column source_strings,
          size_type width,
          fill_char):
    """
    Returns a Column by filling left side of strings in `source_strings`
    with additional character, `fill_char` up to the given `width`.
    """
    return pad(source_strings, width, fill_char, plc.strings.side_type.SideType.LEFT)
