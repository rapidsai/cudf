# Copyright (c) 2018-2024, NVIDIA CORPORATION.

from cudf.core.buffer import acquire_spill_lock

from cudf._lib.column cimport Column

from cudf._lib.pylibcudf.strings import case


@acquire_spill_lock()
def to_upper(Column source_strings):
    return Column.from_pylibcudf(
            case.to_upper(
                source_strings.to_pylibcudf(mode='read')
            )
    )


@acquire_spill_lock()
def to_lower(Column source_strings):
    return Column.from_pylibcudf(
            case.to_lower(
                source_strings.to_pylibcudf(mode='read')
            )
    )


@acquire_spill_lock()
def swapcase(Column source_strings):
    return Column.from_pylibcudf(
            case.swapcase(
                source_strings.to_pylibcudf(mode='read')
            )
    )
