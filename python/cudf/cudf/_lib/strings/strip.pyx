# Copyright (c) 2020-2024, NVIDIA CORPORATION.

from cudf.core.buffer import acquire_spill_lock

from cudf._lib.column cimport Column
import pylibcudf as plc


@acquire_spill_lock()
def strip(Column source_strings,
          object py_repl):
    """
    Returns a Column by removing leading and trailing characters.
    The set of characters need be stripped from left and right side
    can be specified by `py_repl`.
    """
    plc_result = plc.strings.strip.strip(
        source_strings.to_pylibcudf(mode="read"),
        plc.strings.side_type.SideType.BOTH,
        py_repl.device_value.c_value,
    )
    return Column.from_pylibcudf(plc_result)


@acquire_spill_lock()
def lstrip(Column source_strings,
           object py_repl):
    """
    Returns a Column by removing leading and trailing characters.
    The set of characters need be stripped from left side can
    be specified by `py_repl`.
    """
    plc_result = plc.strings.strip.strip(
        source_strings.to_pylibcudf(mode="read"),
        plc.strings.side_type.SideType.LEFT,
        py_repl.device_value.c_value,
    )
    return Column.from_pylibcudf(plc_result)


@acquire_spill_lock()
def rstrip(Column source_strings,
           object py_repl):
    """
    Returns a Column by removing leading and trailing characters.
    The set of characters need be stripped from right side can
    be specified by `py_repl`.
    """
    plc_result = plc.strings.strip.strip(
        source_strings.to_pylibcudf(mode="read"),
        plc.strings.side_type.SideType.RIGHT,
        py_repl.device_value.c_value,
    )
    return Column.from_pylibcudf(plc_result)
