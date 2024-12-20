# Copyright (c) 2023-2024, NVIDIA CORPORATION.

from cudf.core.buffer import acquire_spill_lock

from cudf._lib.column cimport Column

from pylibcudf import nvtext


@acquire_spill_lock()
def jaccard_index(Column input1, Column input2, int width):
    result = nvtext.jaccard.jaccard_index(
        input1.to_pylibcudf(mode="read"),
        input2.to_pylibcudf(mode="read"),
        width,
    )
    return Column.from_pylibcudf(result)
