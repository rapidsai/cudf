# Copyright (c) 2020-2024, NVIDIA CORPORATION.

from pylibcudf.libcudf.types cimport size_type
import pylibcudf as plc

from cudf.core.buffer import acquire_spill_lock

from cudf._lib.column cimport Column


@acquire_spill_lock()
def replace_re(Column source_strings,
               object pattern,
               object py_repl,
               size_type n):
    """
    Returns a Column after replacing occurrences regular
    expressions `pattern` with `py_repl` in `source_strings`.
    `n` indicates the number of resplacements to be made from
    start. (-1 indicates all)
    """
    plc_column = plc.strings.replace_re.replace_re(
        source_strings.to_pylibcudf(mode="read"),
        plc.strings.regex_program.RegexProgram.create(
            str(pattern),
            plc.strings.regex_flags.RegexFlags.DEFAULT
        ),
        py_repl.device_value.c_value,
        n
    )
    return Column.from_pylibcudf(plc_column)


@acquire_spill_lock()
def replace_with_backrefs(
        Column source_strings,
        object pattern,
        object repl):
    """
    Returns a Column after using the `repl` back-ref template to create
    new string with the extracted elements found using
    `pattern` regular expression in `source_strings`.
    """
    plc_column = plc.strings.replace_re.replace_with_backrefs(
        source_strings.to_pylibcudf(mode="read"),
        plc.strings.regex_program.RegexProgram.create(
            str(pattern),
            plc.strings.regex_flags.RegexFlags.DEFAULT
        ),
        repl
    )
    return Column.from_pylibcudf(plc_column)


@acquire_spill_lock()
def replace_multi_re(Column source_strings,
                     list patterns,
                     Column repl_strings):
    """
    Returns a Column after replacing occurrences of multiple
    regular expressions `patterns` with their corresponding
    strings in `repl_strings` in `source_strings`.
    """
    plc_column = plc.strings.replace_re.replace_re(
        source_strings.to_pylibcudf(mode="read"),
        patterns,
        repl_strings.to_pylibcudf(mode="read")
    )
    return Column.from_pylibcudf(plc_column)
