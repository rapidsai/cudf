# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Assertions for validating the results cudf-polars benchmarks."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import polars as pl
import polars.testing

from cudf_polars.utils.versions import POLARS_VERSION_LT_1323

if TYPE_CHECKING:
    from typing import Any


class ValidationError(AssertionError):
    """
    Exception raised when validation fails.

    Parameters
    ----------
    message : str
        The message to display when the validation fails.
    details : dict[str, Any] | None, optional
        Additional details about the validation failure.
    """

    def __init__(self, message: str, details: dict[str, Any] | None = None):
        super().__init__(message)
        self.message = message
        self.details = details


def assert_tpch_result_equal(
    left: pl.DataFrame,
    right: pl.DataFrame,
    *,
    check_row_order: bool = True,
    check_column_order: bool = True,
    check_dtypes: bool = True,
    check_exact: bool = True,
    rel_tol: float = 1e-05,
    abs_tol: float = 1e-08,
    categorical_as_str: bool = False,
    sort_by: list[tuple[str, bool]],
    limit: int | None = None,
) -> None:
    """
    Validate the computed result against the expected answer.

    Parameters
    ----------
    left : pl.DataFrame
        The computed result to validate.
    right : pl.DataFrame
        The expected answer to validate against.
    check_row_order, check_column_order, check_dtypes, check_exact, categorical_as_str, rel_tol, abs_tol
        Same meaning as in polars.
    sort_by : list[tuple[str, bool]]
        The columns to sort by, and the sort order. This *must* be the same
        as the ``sort_by`` and ``descending`` required by the query
    limit : int | None, optional
        The limit (passed to ``.head``) used in the query, if any. This is
        used to break ties in the ``sort_by`` columns. See notes below.

    Returns
    -------
    validation_result

    Notes
    -----
    This validates that:

    1. The schema (column names and data types) match
    2. The values match, with some special handling
       - approximate comparison (for floating point values)
       - sorting stability / distributed execution

    Consider a set of ``(key, value)`` records like::

       ("a", 1)
       ("b", 1)
       ("c", 1)
       ("d", 1)

    Now suppose we run a query that sorts on ``value``. *Any* ordering of those
    records is as correct as any other, since the ``value`` is the same and they
    query says nothing about the sorting of the other columns.

    To handle this, this function sorts the result and expected dataframes, taking
    care to sort by the ``sort_by`` columns *first* (preserving the semantics of the
    query) and then by the remaining columns.

    After sorting by all the columns, any remaining differences are should be
    real, *unless* the query includes a ``limit`` / ``.head(n)`` component. Consider
    a query that includes a ``.sort_by("value").head(2)`` component. In our example,
    any result that returns exactly two rows is as good as any other.

    To handle this, this comparison function does the value comparison in two
    parts when there's a ``.sort_by(...).head(n)`` component:

    1. For all the values "before" the last value (defined by ``sort_by``), we
       compare the results directly using ``pl.testing.assert_frame_equal``.
    2. For the "ties", we make sure that the lengths of the two dataframes match,
       but we *don't* compare the values since, aside from the columns in ``sort_by``,
       the values may differ, and that's OK.
    """
    detail: dict[str, Any]

    polars_kwargs: dict[str, bool | float] = {
        "check_row_order": check_row_order,
        "check_column_order": check_column_order,
        "check_dtypes": check_dtypes,
        "check_exact": check_exact,
        "categorical_as_str": categorical_as_str,
    }

    if POLARS_VERSION_LT_1323:  # pragma: no cover
        tol_kwargs = {"rtol": rel_tol, "atol": abs_tol}
    else:
        tol_kwargs = {"rel_tol": rel_tol, "abs_tol": abs_tol}
    polars_kwargs.update(tol_kwargs)

    if left.columns != right.columns:
        extra = set(left.columns) - set(right.columns)
        missing = set(right.columns) - set(left.columns)
        detail = {
            "type": "column_names_mismatch",
            "expected_columns": right.columns,
            "result_columns": left.columns,
            "mismatched_columns": {
                "extra": extra,
                "missing": missing,
            },
        }
        raise ValidationError(message="Column names mismatch", details=detail)

    # Then, check the schema
    if left.schema != right.schema:
        detail = {
            "type": "schema_mismatch",
            "expected_schema": {k: str(v) for k, v in right.schema.items()},
            "result_schema": {k: str(v) for k, v in left.schema.items()},
            "mismatched_columns": [
                {
                    "name": col,
                    "expected_type": str(right.schema[col]),
                    "result_type": str(left.schema[col]),
                }
                for col in left.columns
                if left.schema[col] != right.schema[col]
            ],
        }
        raise ValidationError(message="Schema mismatch", details=detail)

    # For reasons... the polars / cudf-polars Decimal implementation differs
    # slightly from the DuckDB implementation, in ways that can result in *small*
    # but *real* differences in the results (off by 1%).
    float_casts = [
        pl.col(col).cast(pl.Float64())
        for col in left.columns
        if left.schema[col].is_decimal()
    ]
    right = right.with_columns(*float_casts)
    left = left.with_columns(*float_casts)

    if sort_by:
        sort_by_cols, sort_by_descending = zip(*sort_by, strict=True)

        # Before we do any sorting, we want to verify that the `sort_by` columns match exactly.
        try:
            polars.testing.assert_frame_equal(
                left.select(sort_by_cols),
                right.select(sort_by_cols),
                **polars_kwargs,  # type: ignore[arg-type]
            )
        except AssertionError as e:
            raise ValidationError(
                message="sort_by columns mismatch", details={"error": str(e)}
            ) from e

    else:
        sort_by_cols = ()
        sort_by_descending = ()

    if sort_by and limit and len(left) > 0:
        # Handle the .sort_by(...).head(n) case; First, split the data into two parts
        # "before" and "ties"
        # Problem: suppose we're splitting on a float column: small, floating-point precision
        # differences, which we'd ignore in assert_frame_equal, might cause us to
        # split the two dataframes at different, but not meaningfully different, points.
        # Suppose you have:
        #
        # result  : [a, b, c, d, d+epsilon]
        # expected: [a, b, c, d-epsilon, d]
        #
        # For epsilon less than our tolerance, we'd want to consider this valid.
        # Meaning that the split point should be `d+epsilon` and the
        # partitions should be
        # result:   [ [a, b, c], [d, d + epsilon] ]
        # expected: [ [a, b, c], [d - epsilon, d] ]

        sort_by_descending_list = list(sort_by_descending)
        (split_at,) = (
            left.select(sort_by_cols)
            .sort(by=sort_by_cols, descending=sort_by_descending_list)
            .tail(1)
            .to_dicts()
        )
        # Note that we multiply abs_tol by 2; In our example above, our split point will
        # be d + epsilon; but we want to consider d - epsilon tied to the "real" split point
        # of 'd' as well.

        exprs = []
        for (col, val), desc in zip(
            split_at.items(), sort_by_descending_list, strict=True
        ):
            if isinstance(val, float):
                exprs.append(
                    pl.col(col).lt(val - 2 * abs_tol)
                    | pl.col(col).gt(val + 2 * abs_tol)
                )
            else:
                if desc:
                    # then "before" means "greater than"
                    op = pl.col(col).gt
                else:
                    op = pl.col(col).lt
                exprs.append(op(val))

        expr = pl.Expr.or_(*exprs)

        result_first = left.filter(expr)
        expected_first = right.filter(expr)

        # Before we compare, we need to sort the result and expected.
        # We need to sort by *all* the columns, starting with the
        # columns in `sort_by`; We don't care about the sort order of the remaining
        # columns, just that they're in the same order.
        by = list(sort_by_cols) + [
            col for col in left.columns if col not in sort_by_cols
        ]
        descending = list(sort_by_descending) + [False] * (
            len(left.columns) - len(sort_by_cols)
        )

        result_first = result_first.sort(by=by, descending=descending)
        expected_first = expected_first.sort(by=by, descending=descending)

        sort_by_cols_list = list(sort_by_cols)
        has_float_sort = any(
            left.schema[col] in (pl.Float32, pl.Float64) for col in sort_by_cols_list
        )

        if not has_float_sort:
            try:
                polars.testing.assert_frame_equal(
                    result_first,
                    expected_first,
                    **polars_kwargs,  # type: ignore[arg-type]
                )
            except AssertionError as e:
                raise ValidationError(
                    message="Result mismatch in non-ties part",
                    details={
                        "error": str(e),
                        "split_at": str(split_at),
                    },
                ) from e
        else:
            try:
                _compare_sorted_frames_by_float_bands(
                    result_first,
                    expected_first,
                    sort_by_cols_list=sort_by_cols_list,
                    abs_tol=abs_tol,
                    polars_kwargs=polars_kwargs,
                )
            except ValidationError as e:
                raise ValidationError(
                    message="Result mismatch in non-ties part",
                    details={**(e.details or {}), "split_at": str(split_at)},
                ) from e

        # Now for the ties:
        result_ties = left.filter(~expr)
        expected_ties = right.filter(~expr)

        # We already know that
        # 1. the schema matches (checked above)
        # 2. the values in ``sort_by`` match (else the Expr above would be False)
        # so all that's left to check is that the lengths match.
        if len(result_ties) != len(expected_ties):  # pragma: no cover
            # This *should* be unreachable... We've already checked that the
            # lengths of the two full dataframes match and that the lengths
            # of the two "ties" portions match, so the non-ties portion
            # must match too.
            # But we'll check just in case.
            raise ValidationError(
                message="Ties length mismatch",
                details={
                    "expected_length": len(expected_ties),
                    "result_length": len(result_ties),
                    "split_at": str(split_at),
                },
            )
    else:
        # No limit: sort then compare. If any sort column is float, use
        # band-based comparison so row order within float ties is not required.
        by = list(sort_by_cols) + [
            col for col in left.columns if col not in sort_by_cols
        ]
        descending = list(sort_by_descending) + [False] * (
            len(left.columns) - len(sort_by_cols)
        )

        left_sorted = left.sort(by=by, descending=descending)
        right_sorted = right.sort(by=by, descending=descending)

        sort_by_cols_list = list(sort_by_cols)
        has_float_sort = any(
            left_sorted.schema[col] in (pl.Float32, pl.Float64)
            for col in sort_by_cols_list
        )

        if not has_float_sort:
            try:
                polars.testing.assert_frame_equal(
                    left_sorted,
                    right_sorted,
                    **polars_kwargs,  # type: ignore[arg-type]
                )
            except AssertionError as e:
                raise ValidationError(
                    message="Result mismatch", details={"error": str(e)}
                ) from e
        else:
            # Float-sort path: partition into tie bands by canonical sort key
            # (float columns rounded to abs_tol grid), then compare each band
            # as multiset (sort by non-sort columns, then assert_frame_equal).
            _compare_sorted_frames_by_float_bands(
                left_sorted,
                right_sorted,
                sort_by_cols_list=sort_by_cols_list,
                abs_tol=abs_tol,
                polars_kwargs=polars_kwargs,
            )

    return None


def _compare_sorted_frames_by_float_bands(
    left: pl.DataFrame,
    right: pl.DataFrame,
    *,
    sort_by_cols_list: list[str],
    abs_tol: float,
    polars_kwargs: dict[str, bool | float],
) -> None:
    """
    Compare two already-sorted frames by partitioning into bands on sort key.

    Float sort columns are bucketed by (x / abs_tol).round() * abs_tol so that
    values within tolerance get the same band. Bands are compared in order;
    within each band, rows are sorted by non-sort columns then compared.
    """
    assert len(left) == len(right), "Row count mismatch"

    canonical_exprs: list[pl.Expr] = []
    canonical_col_names: list[str] = []
    for col in sort_by_cols_list:
        name = f"_band_canonical_{col}"
        canonical_col_names.append(name)
        if left.schema[col] in (pl.Float32, pl.Float64):
            canonical_exprs.append(
                ((pl.col(col) / abs_tol).round() * abs_tol).alias(name)
            )
        else:
            canonical_exprs.append(pl.col(col).alias(name))

    def add_band_id(df: pl.DataFrame) -> pl.DataFrame:
        return df.with_columns(canonical_exprs).with_columns(
            pl.struct(canonical_col_names).rle_id().alias("_band_id")
        )

    left_banded = add_band_id(left)
    right_banded = add_band_id(right)

    # max() can be time or int, but we know it's int.
    n_bands_left = cast(int, left_banded["_band_id"].max()) + 1
    n_bands_right = cast(int, right_banded["_band_id"].max()) + 1
    assert n_bands_left == n_bands_right, "Band count mismatch"

    other_cols = [c for c in left.columns if c not in sort_by_cols_list]
    drop_cols = ["_band_id", *canonical_col_names]

    for band_id in range(n_bands_left):
        left_band = left_banded.filter(pl.col("_band_id") == band_id).drop(drop_cols)
        right_band = right_banded.filter(pl.col("_band_id") == band_id).drop(drop_cols)
        if other_cols:
            left_band = left_band.sort(by=other_cols)
            right_band = right_band.sort(by=other_cols)
        assert len(left_band) == len(right_band), "Band length mismatch"
        try:
            polars.testing.assert_frame_equal(
                left_band,
                right_band,
                **polars_kwargs,  # type: ignore[arg-type]
            )
        except AssertionError as e:
            raise ValidationError(
                message="Result mismatch",
                details={"error": str(e), "band_id": band_id},
            ) from e
