# Copyright (c) 2020-2024, NVIDIA CORPORATION.

from cudf.core.buffer import acquire_spill_lock

from pylibcudf.libcudf.types cimport size_type

from cudf._lib.column cimport Column

import pylibcudf as plc


@acquire_spill_lock()
def wrap(Column source_strings,
         size_type width):
    """
    Returns a Column by wrapping long strings
    in the Column to be formatted in paragraphs
    with length less than a given `width`.
    """
    plc_result = plc.strings.wrap.wrap(
        source_strings.to_pylibcudf(mode="read"),
        width
    )
    return Column.from_pylibcudf(plc_result)
