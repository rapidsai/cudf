# Copyright (c) 2023, NVIDIA CORPORATION.
from __future__ import annotations

import functools
from typing import Tuple, cast

import numpy as np

import cudf.core.column as column
from cudf.core.dtypes import is_categorical_dtype


@functools.cache
def common_numeric_type(ltype: np.dtype, rtype: np.dtype) -> np.dtype | None:
    """Return the common type between ltype and rtype or None.

    If there exists a common type it is safe to losslessly promote the
    provided pair of dtype to, it is returned. If there is no safe
    promotion route, None is returned, and it is up to the caller to
    decide how to proceed.
    """
    kinds = {ltype.kind, rtype.kind}
    assert kinds.issubset({"i", "u", "f", "c", "b", "m", "M"})
    # integer
    if kinds.issubset({"i", "u"}):
        # Direct promotion routes for integer types
        # +----+    +-----+    +-----+    +-----+
        # | i8 |--->| i16 |--->| i32 |--->| i64 |
        # +----+    +--^--+    +--^--+    +--^--+
        #             /          /          /
        #         /---'      /---'      /---'
        #     /---'      /---'      /---'
        # +--+-+    +---+-+    +---+-+    +-----+
        # | u8 |--->| u16 |--->| u32 |--->| u64 |
        # +----+    +-----+    +-----+    +-----+
        candidate = np.promote_types(ltype, rtype)
        if candidate.kind in {"i", "u"}:
            return candidate
        else:
            # Promotion to float types not allowed
            return None
    # float/complex
    if kinds.issubset({"f", "c"}):
        # Direct promotion routes for float types
        #            +-----+    +------+    +------+
        #            | c64 |--->| c128 |--->| c256 |
        #            +-----+    +------+    +------+
        #               ^          ^           ^
        #               |          |           |
        #               |          |           |
        # +-----+    +-----+    +-----+     +------+
        # | f16 |--->| f32 |--->| f64 |---->| f128 |
        # +-----+    +-----+    +-----+     +------+
        return np.promote_types(ltype, rtype)
    # These are belt-and-braces since match_join_types checking for
    # dtype equality early means we should succeed in these checks,
    # but we encode the rules in case someone else wants to use them.
    # bool
    if kinds.issubset({"b"}):
        return ltype
    # datetime
    if kinds.issubset({"m", "M"}):
        # Promotion between mixed-resolution datetimes is not
        # guaranteed safe without inspecting the values, since the
        # range of a datetime[M] is much larger than a datetime[ns].
        # Pandas does value-dependent cast or raise. We will just
        # raise. Similarly, we're not allowed to promote between
        # datetime and timedelta which is handled by the equality
        # check here.
        if ltype == rtype:
            return ltype
        else:
            return None
    # Nothing else is safe
    return None


def match_join_types(
    left: column.ColumnBase, right: column.ColumnBase
) -> Tuple[column.ColumnBase, column.ColumnBase]:
    """Given a pair of columns to join, return a new pair with
    matching dtypes

    Parameters
    ----------
    left
        Left column to join on
    right
        Right column to join on

    Returns
    -------
    tuple
        Pair of, possibly type-promoted, left and right columns with
        matching dtype

    Raises
    ------
    ValueError
        If there exists no safe promotion rule for the pair of
        columns.

    Notes
    -----
    Non-decimal numeric types are promoted according to the table provided by
    :func:`numeric_promotions` which only allows safe promotions and
    never promotes between numeric kinds. If exactly one column is categorical,
    it is decategorized and promotion continues with the decategorized
    column.

    All other dtypes must match exactly, so there is no automatic
    promotion between (for example) decimal columns of different precision.
    """
    ltype = left.dtype
    rtype = right.dtype

    if ltype == rtype:
        return left, right
    left_is_cat = is_categorical_dtype(left.dtype)
    right_is_cat = is_categorical_dtype(right.dtype)

    # If categorical dtypes don't match exactly, decategorize and try
    # matching those.
    if left_is_cat and right_is_cat:
        return match_join_types(
            cast(column.CategoricalColumn, left)._get_decategorized_column(),
            cast(column.CategoricalColumn, right)._get_decategorized_column(),
        )
    elif left_is_cat:
        return match_join_types(
            cast(column.CategoricalColumn, left)._get_decategorized_column(),
            right,
        )
    elif right_is_cat:
        return match_join_types(
            left,
            cast(column.CategoricalColumn, right)._get_decategorized_column(),
        )
    elif ltype.kind == "O" or rtype.kind == "O":
        # Categorical columns also have kind == "O" but are handled
        # explicitly above.
        raise ValueError(
            f"Cannot merge on non-matching key types {ltype} and {rtype}"
        )
    elif (
        common_type := common_numeric_type(
            cast(np.dtype, ltype), cast(np.dtype, rtype)
        )
    ) is not None:
        # Numpy-supported dtype which we can safely promote
        return left.astype(common_type), right.astype(common_type)
    else:
        # Numpy-supported dtype which we cannot safely promote
        raise ValueError(
            f"Cannot safely promote numeric pair {ltype} and {rtype}. "
            "To perform the merge, manually cast the merge keys to "
            "an appropriate common type first."
        )
