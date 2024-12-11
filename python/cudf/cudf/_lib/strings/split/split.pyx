# Copyright (c) 2020-2024, NVIDIA CORPORATION.

from cudf.core.buffer import acquire_spill_lock

from pylibcudf.libcudf.types cimport size_type

from cudf._lib.column cimport Column

import pylibcudf as plc


@acquire_spill_lock()
def split(Column source_strings,
          object py_delimiter,
          size_type maxsplit):
    """
    Returns data by splitting the `source_strings`
    column around the specified `py_delimiter`.
    The split happens from beginning.
    """
    plc_table = plc.strings.split.split.split(
        source_strings.to_pylibcudf(mode="read"),
        py_delimiter.device_value.c_value,
        maxsplit,
    )
    return dict(enumerate(Column.from_pylibcudf(col) for col in plc_table.columns()))


@acquire_spill_lock()
def split_record(Column source_strings,
                 object py_delimiter,
                 size_type maxsplit):
    """
    Returns a Column by splitting the `source_strings`
    column around the specified `py_delimiter`.
    The split happens from beginning.
    """
    plc_column = plc.strings.split.split.split_record(
        source_strings.to_pylibcudf(mode="read"),
        py_delimiter.device_value.c_value,
        maxsplit,
    )
    return Column.from_pylibcudf(plc_column)


@acquire_spill_lock()
def rsplit(Column source_strings,
           object py_delimiter,
           size_type maxsplit):
    """
    Returns data by splitting the `source_strings`
    column around the specified `py_delimiter`.
    The split happens from the end.
    """
    plc_table = plc.strings.split.split.rsplit(
        source_strings.to_pylibcudf(mode="read"),
        py_delimiter.device_value.c_value,
        maxsplit,
    )
    return dict(enumerate(Column.from_pylibcudf(col) for col in plc_table.columns()))


@acquire_spill_lock()
def rsplit_record(Column source_strings,
                  object py_delimiter,
                  size_type maxsplit):
    """
    Returns a Column by splitting the `source_strings`
    column around the specified `py_delimiter`.
    The split happens from the end.
    """
    plc_column = plc.strings.split.split.rsplit_record(
        source_strings.to_pylibcudf(mode="read"),
        py_delimiter.device_value.c_value,
        maxsplit,
    )
    return Column.from_pylibcudf(plc_column)


@acquire_spill_lock()
def split_re(Column source_strings,
             object pattern,
             size_type maxsplit):
    """
    Returns data by splitting the `source_strings`
    column around the delimiters identified by `pattern`.
    """
    plc_table = plc.strings.split.split.split_re(
        source_strings.to_pylibcudf(mode="read"),
        plc.strings.regex_program.RegexProgram.create(
            str(pattern),
            plc.strings.regex_flags.RegexFlags.DEFAULT,
        ),
        maxsplit,
    )
    return dict(enumerate(Column.from_pylibcudf(col) for col in plc_table.columns()))


@acquire_spill_lock()
def rsplit_re(Column source_strings,
              object pattern,
              size_type maxsplit):
    """
    Returns data by splitting the `source_strings`
    column around the delimiters identified by `pattern`.
    The delimiters are searched starting from the end of each string.
    """
    plc_table = plc.strings.split.split.rsplit_re(
        source_strings.to_pylibcudf(mode="read"),
        plc.strings.regex_program.RegexProgram.create(
            str(pattern),
            plc.strings.regex_flags.RegexFlags.DEFAULT,
        ),
        maxsplit,
    )
    return dict(enumerate(Column.from_pylibcudf(col) for col in plc_table.columns()))


@acquire_spill_lock()
def split_record_re(Column source_strings,
                    object pattern,
                    size_type maxsplit):
    """
    Returns a Column by splitting the `source_strings`
    column around the delimiters identified by `pattern`.
    """
    plc_column = plc.strings.split.split.split_record_re(
        source_strings.to_pylibcudf(mode="read"),
        plc.strings.regex_program.RegexProgram.create(
            str(pattern),
            plc.strings.regex_flags.RegexFlags.DEFAULT,
        ),
        maxsplit,
    )
    return Column.from_pylibcudf(plc_column)


@acquire_spill_lock()
def rsplit_record_re(Column source_strings,
                     object pattern,
                     size_type maxsplit):
    """
    Returns a Column by splitting the `source_strings`
    column around the delimiters identified by `pattern`.
    The delimiters are searched starting from the end of each string.
    """
    plc_column = plc.strings.split.split.rsplit_record_re(
        source_strings.to_pylibcudf(mode="read"),
        plc.strings.regex_program.RegexProgram.create(
            str(pattern),
            plc.strings.regex_flags.RegexFlags.DEFAULT,
        ),
        maxsplit,
    )
    return Column.from_pylibcudf(plc_column)
