# Copyright (c) 2020-2024, NVIDIA CORPORATION.

from libcpp cimport bool

from cudf._lib.utils cimport columns_from_pylibcudf_table

from cudf._lib import pylibcudf


def merge_sorted(
    list input_columns,
    list key_columns_indices,
    bool ascending=True,
    str na_position="last",
):
    """Merge multiple lists of lexicographically sorted columns into one list
    of sorted columns. `input_columns` is a list of lists of columns to be
    merged.
    """
    c_input_tables = [
        pylibcudf.Table(
            [c.to_pylibcudf(mode="read") for c in source_columns]
        ) for source_columns in input_columns
    ]

    num_keys = len(key_columns_indices)

    column_order = (
        pylibcudf.types.Order.ASCENDING if ascending
        else pylibcudf.types.Order.DESCENDING
    )

    if not ascending:
        na_position = "last" if na_position == "first" else "first"
    null_precedence = (
        pylibcudf.types.NullOrder.BEFORE if na_position == "first"
        else pylibcudf.types.NullOrder.AFTER
    )

    return columns_from_pylibcudf_table(
        pylibcudf.merge.merge(
            c_input_tables,
            key_columns_indices,
            [column_order] * num_keys,
            [null_precedence] * num_keys,
        )
    )
