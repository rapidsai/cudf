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
    # TODO(s) in the order specified :
    # 1. The return type needs to be changed
    #    to cudf.Categorical once it is implemented.
    # 2. Make this API public (i.e., to resemble
    #    pd.api.types.union_categoricals)

    if ignore_order:
        raise TypeError("ignore_order is not yet implemented")

    result_col = cudf.core.column.CategoricalColumn._concat(
        [obj._column for obj in to_union]
    )
    if sort_categories:
        sorted_categories = result_col.categories.sort_by_values(
            ascending=True
        )[0]
        result_col = result_col.cat().reorder_categories(
            new_categories=sorted_categories
        )

    return cudf.Index(result_col)
