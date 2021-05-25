# Copyright (c) 2021, NVIDIA CORPORATION.

from __future__ import annotations

from typing import List, Union

import cudf

ParentType = Union["cudf.Series", "cudf.Index"]


def _union_categoricals(
    to_union: List[Union[cudf.Series, cudf.Index]],
    sort_categories: bool = False,
    ignore_order: bool = False,
):
    """
    This is an internal API which combines categorical data.
    """

    if ignore_order:
        raise TypeError("ignore_order is not yet implemented")

    result_col = cudf.core.column.CategoricalColumn._concat(
        [obj._column for obj in to_union]
    )
    if sort_categories:
        sorted_categories = (
            cudf.Series(result_col.categories)
            .sort_values(ascending=True, ignore_index=True)
            ._column
        )
        result_col = result_col.cat().reorder_categories(
            new_categories=sorted_categories
        )

    # TODO: The return type needs to be changed
    # to cudf.Categorical once it is implemented.

    return cudf.Index(result_col)
