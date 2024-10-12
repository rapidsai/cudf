# Copyright (c) 2018-2024, NVIDIA CORPORATION.

from cudf.core.buffer import acquire_spill_lock

from libcpp cimport bool
from libcpp.memory cimport unique_ptr
from libcpp.utility cimport move

from pylibcudf.libcudf.column.column cimport column
from pylibcudf.libcudf.column.column_view cimport column_view
from pylibcudf.libcudf.nvtext.normalize cimport (
    normalize_characters as cpp_normalize_characters,
    normalize_spaces as cpp_normalize_spaces,
)

from cudf._lib.column cimport Column

from pylibcudf import nvtext


@acquire_spill_lock()
def normalize_spaces(Column input):
    result = nvtext.normalize.normalize_spaces(
        input.to_pylibcudf(mode="read")
    )
    return Column.from_pylibcudf(result)


@acquire_spill_lock()
def normalize_characters(Column input, bool do_lower=True):
    result = nvtext.normalize.normalize_characters(
        input.to_pylibcudf(mode="read"),
        do_lower,
    )
    return Column.from_pylibcudf(result)
