# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Assertions for validating the results cudf-polars benchmarks."""

from __future__ import annotations

from typing import TYPE_CHECKING

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

    if sort_by and limit:
        # Handle the .sort_by(...).head(n) case; First, split the data into two parts
        # "before" and "ties"
        sort_by_cols, sort_by_descending = zip(*sort_by, strict=True)

        # Problem: suppose we're splitting on a float column: small, floating-point precision
        # differences, which we'd ignore in assert_frame_equal, might cause us to
        # split the two dataframes at different, but not meaningfully different, points.
        # Suppose you have:
        #
        # result  : [a, b, c, d, d+epsilon]
        # expected: [a, b, c, d-epsilon, d]
        #
        # For epsilon less than our tolerance, we'd want to consider this valid.

        # Use the lexicographic last row in the query's sort order (ORDER BY ...).
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
        #
        # "Strictly before" in the sort order: for each column, ascending means "before" =
        # less than, descending means "before" = greater than. Build lexicographic
        # "row < split_at" as: cond_0 or (eq_0 and cond_1) or (eq_0 and eq_1 and cond_2) ...

        exprs = []
        for (col, val), desc in zip(
            split_at.items(), sort_by_descending_list, strict=True
        ):
            if isinstance(val, float):
                # eq = (pl.col(col) >= val - 2 * abs_tol) & (pl.col(col) <= val + 2 * abs_tol)
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

        # validate this part normally:
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

        left = left.sort(by=by, descending=descending)
        right = right.sort(by=by, descending=descending)

        try:
            polars.testing.assert_frame_equal(left, right, **polars_kwargs)  # type: ignore[arg-type]
        except AssertionError as e:
            raise ValidationError(
                message="Result mismatch", details={"error": str(e)}
            ) from e

    return None
