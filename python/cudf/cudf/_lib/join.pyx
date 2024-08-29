# Copyright (c) 2020-2024, NVIDIA CORPORATION.

from cudf.core.buffer import acquire_spill_lock

from cudf._lib.column cimport Column

import pylibcudf

# The functions below return the *gathermaps* that represent
# the join result when joining on the keys `lhs` and `rhs`.


@acquire_spill_lock()
def join(list lhs, list rhs, how=None):
    if how == "outer":
        how = "full"
    if (join_func := getattr(pylibcudf.join, f"{how}_join", None)) is None:
        raise ValueError(f"Invalid join type {how}")

    left_rows, right_rows = join_func(
        pylibcudf.Table([c.to_pylibcudf(mode="read") for c in lhs]),
        pylibcudf.Table([c.to_pylibcudf(mode="read") for c in rhs]),
        pylibcudf.types.NullEquality.EQUAL
    )
    return Column.from_pylibcudf(left_rows), Column.from_pylibcudf(right_rows)


@acquire_spill_lock()
def semi_join(list lhs, list rhs, how=None):
    if (
        join_func := getattr(
            pylibcudf.join, f"{how.replace('left', 'left_')}_join", None
        )
    ) is None:
        raise ValueError(f"Invalid join type {how}")

    return Column.from_pylibcudf(
        join_func(
            pylibcudf.Table([c.to_pylibcudf(mode="read") for c in lhs]),
            pylibcudf.Table([c.to_pylibcudf(mode="read") for c in rhs]),
            pylibcudf.types.NullEquality.EQUAL
        )
    ), None
