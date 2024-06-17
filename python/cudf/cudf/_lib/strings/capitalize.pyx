# Copyright (c) 2018-2024, NVIDIA CORPORATION.

from cudf.core.buffer import acquire_spill_lock

from cudf._lib.column cimport Column

import cudf._lib.pylibcudf as plc


@acquire_spill_lock()
def capitalize(Column source_strings):
    return Column.from_pylibcudf(
        plc.strings.capitalize.capitalize(
            source_strings.to_pylibcudf(mode="read")
        )
    )


@acquire_spill_lock()
def title(Column source_strings):
    return Column.from_pylibcudf(
        plc.strings.capitalize.title(
            source_strings.to_pylibcudf(mode="read")
        )
    )


@acquire_spill_lock()
def is_title(Column source_strings):
    return Column.from_pylibcudf(
        plc.strings.capitalize.is_title(
            source_strings.to_pylibcudf(mode="read")
        )
    )
