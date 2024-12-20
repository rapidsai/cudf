# Copyright (c) 2018-2024, NVIDIA CORPORATION.

from libcpp cimport bool

from cudf.core.buffer import acquire_spill_lock

from cudf._lib.column cimport Column

import pylibcudf as plc


@acquire_spill_lock()
def translate(Column source_strings,
              object mapping_table):
    """
    Translates individual characters within each string
    if present in the mapping_table.
    """
    plc_result = plc.strings.translate.translate(
        source_strings.to_pylibcudf(mode="read"),
        mapping_table,
    )
    return Column.from_pylibcudf(plc_result)


@acquire_spill_lock()
def filter_characters(Column source_strings,
                      object mapping_table,
                      bool keep,
                      object py_repl):
    """
    Removes or keeps individual characters within each string
    using the provided mapping_table.
    """
    plc_result = plc.strings.translate.filter_characters(
        source_strings.to_pylibcudf(mode="read"),
        mapping_table,
        plc.strings.translate.FilterType.KEEP
        if keep else plc.strings.translate.FilterType.REMOVE,
        py_repl.device_value.c_value
    )
    return Column.from_pylibcudf(plc_result)
