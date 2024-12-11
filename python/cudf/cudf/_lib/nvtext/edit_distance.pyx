# Copyright (c) 2020-2024, NVIDIA CORPORATION.

from cudf.core.buffer import acquire_spill_lock

from pylibcudf cimport nvtext

from cudf._lib.column cimport Column


@acquire_spill_lock()
def edit_distance(Column strings, Column targets):
    result = nvtext.edit_distance.edit_distance(
        strings.to_pylibcudf(mode="read"),
        targets.to_pylibcudf(mode="read")
    )
    return Column.from_pylibcudf(result)


@acquire_spill_lock()
def edit_distance_matrix(Column strings):
    result = nvtext.edit_distance.edit_distance_matrix(
        strings.to_pylibcudf(mode="read")
    )
    return Column.from_pylibcudf(result)
