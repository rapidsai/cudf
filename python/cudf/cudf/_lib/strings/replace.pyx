# Copyright (c) 2020-2024, NVIDIA CORPORATION.

from libc.stdint cimport int32_t

from cudf.core.buffer import acquire_spill_lock

from pylibcudf.libcudf.types cimport size_type

from cudf._lib.column cimport Column
from cudf._lib.scalar cimport DeviceScalar

import pylibcudf as plc


@acquire_spill_lock()
def slice_replace(Column source_strings,
                  size_type start,
                  size_type stop,
                  object py_repl):
    """
    Returns a Column by replacing specified section
    of each string with `py_repl`. Positions can be
    specified with `start` and `stop` params.
    """

    cdef DeviceScalar repl = py_repl.device_value

    return Column.from_pylibcudf(plc.strings.replace.replace_slice(
        source_strings.to_pylibcudf(mode="read"),
        repl.c_value,
        start,
        stop
    ))


@acquire_spill_lock()
def insert(Column source_strings,
           size_type start,
           object py_repl):
    """
    Returns a Column by inserting a specified
    string `py_repl` at a specific position in all strings.
    """

    cdef DeviceScalar repl = py_repl.device_value

    return Column.from_pylibcudf(plc.strings.replace.replace_slice(
        source_strings.to_pylibcudf(mode="read"),
        repl.c_value,
        start,
        start,
    ))


@acquire_spill_lock()
def replace(Column source_strings,
            object py_target,
            object py_repl,
            int32_t maxrepl):
    """
    Returns a Column after replacing occurrences of
    patterns `py_target` with `py_repl` in `source_strings`.
    `maxrepl` indicates number of replacements to make from start.
    """
    cdef DeviceScalar target = py_target.device_value
    cdef DeviceScalar repl = py_repl.device_value

    return Column.from_pylibcudf(plc.strings.replace.replace(
        source_strings.to_pylibcudf(mode="read"),
        target.c_value,
        repl.c_value,
        maxrepl
    ))


@acquire_spill_lock()
def replace_multi(Column source_strings,
                  Column target_strings,
                  Column repl_strings):
    """
    Returns a Column after replacing occurrences of
    patterns `target_strings` with `repl_strings` in `source_strings`.
    """
    return Column.from_pylibcudf(plc.strings.replace.replace_multiple(
        source_strings.to_pylibcudf(mode="read"),
        target_strings.to_pylibcudf(mode="read"),
        repl_strings.to_pylibcudf(mode="read"),
    ))
