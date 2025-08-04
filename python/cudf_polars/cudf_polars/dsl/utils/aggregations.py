# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Utilities for rewriting aggregations."""

from __future__ import annotations

import itertools
from functools import partial
from typing import TYPE_CHECKING, Any

import polars as pl

import pylibcudf as plc

from cudf_polars.containers import DataType
from cudf_polars.dsl import expr, ir

if TYPE_CHECKING:
    from collections.abc import Callable, Generator, Iterable, Sequence

    from cudf_polars.typing import Schema

__all__ = ["apply_pre_evaluation", "decompose_aggs", "decompose_single_agg"]


def replace_nulls(col: expr.Expr, value: Any, *, is_top: bool) -> expr.Expr:
    """
    Replace nulls with the given scalar if at top level.

    Parameters
    ----------
    col
        Expression to replace nulls in.
    value
        Scalar replacement
    is_top
        Is this top-level (should replacement be performed).

    Returns
    -------
    Massaged expression.
    """
    if not is_top:
        return col
    return expr.UnaryFunction(
        col.dtype, "fill_null", (), col, expr.Literal(col.dtype, value)
    )


def decompose_single_agg(
    named_expr: expr.NamedExpr,
    name_generator: Generator[str, None, None],
    *,
    is_top: bool,
) -> tuple[list[tuple[expr.NamedExpr, bool]], expr.NamedExpr]:
    """
    Decompose a single named aggregation.

    Parameters
    ----------
    named_expr
        The named aggregation to decompose
    name_generator
        Generator of unique names for temporaries introduced during decomposition.
    is_top
        Is this the top of an aggregation expression?

    Returns
    -------
    aggregations
        Pairs of expressions to apply as grouped aggregations (whose children
        may be evaluated pointwise) and flags indicating if the
        expression contained nested aggregations.
    post_aggregate
        Single expression to apply to post-process the grouped
        aggregations.

    Raises
    ------
    NotImplementedError
        If the expression contains nested aggregations or unsupported
        operations in a grouped aggregation context.
    """
    agg = named_expr.value
    name = named_expr.name
    if isinstance(agg, expr.UnaryFunction) and agg.name == "null_count":
        raise NotImplementedError("null_count is not supported inside groupby context")
    if isinstance(agg, expr.Col):
        # TODO: collect_list produces null for empty group in libcudf, empty list in polars.
        # But we need the nested value type, so need to track proper dtypes in our DSL.
        return [(named_expr, False)], named_expr.reconstruct(expr.Col(agg.dtype, name))
    if is_top and isinstance(agg, expr.Cast) and isinstance(agg.children[0], expr.Len):
        # Special case to fill nulls with zeros for empty group length calculations
        (child,) = agg.children
        child_agg, post = decompose_single_agg(
            expr.NamedExpr(next(name_generator), child), name_generator, is_top=True
        )
        return child_agg, named_expr.reconstruct(
            replace_nulls(
                agg.reconstruct([post.value]),
                0,
                is_top=True,
            )
        )
    if isinstance(agg, expr.Len):
        return [(named_expr, True)], named_expr.reconstruct(expr.Col(agg.dtype, name))
    if isinstance(agg, (expr.Literal, expr.LiteralColumn)):
        return [], named_expr
    if isinstance(agg, expr.Agg):
        if agg.name == "quantile":
            # Second child the requested quantile (which is asserted
            # to be a literal on construction)
            child = agg.children[0]
        else:
            (child,) = agg.children
        needs_masking = agg.name in {"min", "max"} and plc.traits.is_floating_point(
            child.dtype.plc
        )
        if needs_masking and agg.options:
            # pl.col("a").nan_max or nan_min
            raise NotImplementedError("Nan propagation in groupby for min/max")
        aggs, _ = decompose_single_agg(
            expr.NamedExpr(next(name_generator), child), name_generator, is_top=False
        )
        if any(has_agg for _, has_agg in aggs):
            raise NotImplementedError("Nested aggs in groupby not supported")
        if needs_masking:
            child = expr.UnaryFunction(child.dtype, "mask_nans", (), child)
            # The aggregation is just reconstructed with the new
            # (potentially masked) child. This is safe because we recursed
            # to ensure there are no nested aggregations.
            return (
                [(named_expr.reconstruct(agg.reconstruct([child])), True)],
                named_expr.reconstruct(expr.Col(agg.dtype, name)),
            )
        elif agg.name == "sum":
            col = (
                expr.Cast(agg.dtype, expr.Col(DataType(pl.datatypes.Int64()), name))
                if (
                    plc.traits.is_integral(agg.dtype.plc)
                    and agg.dtype.id() != plc.TypeId.INT64
                )
                else expr.Col(agg.dtype, name)
            )
            return [(named_expr, True)], expr.NamedExpr(
                name,
                # In polars sum(empty_group) => 0, but in libcudf
                # sum(empty_group) => null So must post-process by
                # replacing nulls, but only if we're a "top-level"
                # agg.
                replace_nulls(col, 0, is_top=is_top),
            )
        elif agg.name == "mean":
            post_agg_col: expr.Expr = expr.Col(
                DataType(pl.Float64), name
            )  # libcudf promotes to float64
            if agg.dtype.plc.id() == plc.TypeId.FLOAT32:
                # Cast back to float32 to match Polars
                post_agg_col = expr.Cast(agg.dtype, post_agg_col)
            return [(named_expr, True)], named_expr.reconstruct(post_agg_col)
        else:
            return [(named_expr, True)], named_expr.reconstruct(
                expr.Col(agg.dtype, name)
            )
    if isinstance(agg, expr.Ternary):
        raise NotImplementedError("Ternary inside groupby")
    if not agg.is_pointwise and isinstance(agg, expr.BooleanFunction):
        raise NotImplementedError(
            f"Non pointwise boolean function {agg.name!r} not supported in groupby or rolling context"
        )
    if agg.is_pointwise:
        aggs, posts = _decompose_aggs(
            (expr.NamedExpr(next(name_generator), child) for child in agg.children),
            name_generator,
            is_top=False,
        )
        if any(has_agg for _, has_agg in aggs):
            if not all(
                has_agg or isinstance(agg.value, expr.Literal) for agg, has_agg in aggs
            ):
                raise NotImplementedError(
                    "Broadcasting aggregated expressions in groupby/rolling"
                )
            # Any pointwise expression can be handled either by
            # post-evaluation (if outside an aggregation).
            return (
                aggs,
                named_expr.reconstruct(agg.reconstruct([p.value for p in posts])),
            )
        else:
            # Or pre-evaluation if inside an aggregation.
            return (
                [(named_expr, False)],
                named_expr.reconstruct(expr.Col(agg.dtype, name)),
            )
    raise NotImplementedError(f"No support for {type(agg)} in groupby/rolling")


def _decompose_aggs(
    aggs: Iterable[expr.NamedExpr],
    name_generator: Generator[str, None, None],
    *,
    is_top: bool,
) -> tuple[list[tuple[expr.NamedExpr, bool]], Sequence[expr.NamedExpr]]:
    new_aggs, post = zip(
        *(decompose_single_agg(agg, name_generator, is_top=is_top) for agg in aggs),
        strict=True,
    )
    return list(itertools.chain.from_iterable(new_aggs)), post


def decompose_aggs(
    aggs: Iterable[expr.NamedExpr], name_generator: Generator[str, None, None]
) -> tuple[list[expr.NamedExpr], Sequence[expr.NamedExpr]]:
    """
    Process arbitrary aggregations into a form we can handle in grouped aggregations.

    Parameters
    ----------
    aggs
        List of aggregation expressions
    name_generator
        Generator of unique names for temporaries introduced during decomposition.

    Returns
    -------
    aggregations
        Aggregations to apply in the groupby node.
    post_aggregations
        Expressions to apply after aggregating (as a ``Select``).

    Notes
    -----
    The aggregation expressions are guaranteed to either be
    expressions that can be pointwise evaluated before the groupby
    operation, or aggregations of such expressions.

    Raises
    ------
    NotImplementedError
        For unsupported aggregation combinations.
    """
    new_aggs, post = _decompose_aggs(aggs, name_generator, is_top=True)
    return [agg for agg, _ in new_aggs], post


def apply_pre_evaluation(
    output_schema: Schema,
    keys: Sequence[expr.NamedExpr],
    original_aggs: Sequence[expr.NamedExpr],
    name_generator: Generator[str, None, None],
    *extra_columns: expr.NamedExpr,
) -> tuple[Sequence[expr.NamedExpr], Schema, Callable[[ir.IR], ir.IR]]:
    """
    Apply pre-evaluation to aggregations in a grouped or rolling context.

    Parameters
    ----------
    output_schema
        Schema of the plan node we're rewriting.
    keys
        Grouping keys (may be empty).
    original_aggs
        Aggregation expressions to rewrite.
    name_generator
        Generator of unique names for temporaries introduced during decomposition.
    extra_columns
        Any additional columns to be included in the output (only
        relevant for rolling aggregations). Columns will appear in the
        order `keys, extra_columns, original_aggs`.

    Returns
    -------
    aggregations
        The required aggregations.
    schema
        The new schema of the aggregation node
    post_process
        Function to apply to the aggregation node to apply any
        post-processing.

    Raises
    ------
    NotImplementedError
        If the aggregations are somehow unsupported.
    """
    aggs, post = decompose_aggs(original_aggs, name_generator)
    assert len(post) == len(original_aggs), (
        f"Unexpected number of post-aggs {len(post)=} {len(original_aggs)=}"
    )
    # Order-preserving unique
    aggs = list(dict.fromkeys(aggs).keys())
    if any(not isinstance(e.value, expr.Col) for e in post):
        selection = [
            *(key.reconstruct(expr.Col(key.value.dtype, key.name)) for key in keys),
            *extra_columns,
            *post,
        ]
        inter_schema = {
            e.name: e.value.dtype for e in itertools.chain(keys, extra_columns, aggs)
        }
        return (
            aggs,
            inter_schema,
            partial(ir.Select, output_schema, selection, True),  # noqa: FBT003
        )
    else:
        return aggs, output_schema, lambda inp: inp
