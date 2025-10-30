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
from cudf_polars.dsl.expressions.base import ExecutionContext
from cudf_polars.utils.versions import POLARS_VERSION_LT_1323

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
    context: ExecutionContext,
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
    context
        ExecutionContext in which the aggregation will run.

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
    if isinstance(agg, expr.UnaryFunction) and agg.name in {
        "rank",
        "fill_null_with_strategy",
        "cum_sum",
    }:
        if context != ExecutionContext.WINDOW:
            raise NotImplementedError(
                f"{agg.name} is not supported in groupby or rolling context"
            )
        if agg.name == "fill_null_with_strategy" and (
            strategy := agg.options[0]
        ) not in {"forward", "backward"}:
            raise NotImplementedError(
                f"fill_null({strategy=}) not supported in a groupy or rolling context"
            )
        # Ensure Polars semantics for dtype:
        # - average -> Float64
        # - min/max/dense/ordinal -> IDX_DTYPE (UInt32/UInt64)
        post_col: expr.Expr = expr.Col(agg.dtype, name)
        if agg.name == "rank":
            post_col = expr.Cast(agg.dtype, post_col)

        return [(named_expr, True)], named_expr.reconstruct(post_col)
    if isinstance(agg, expr.UnaryFunction) and agg.name == "null_count":
        (child,) = agg.children

        is_null_bool = expr.BooleanFunction(
            DataType(pl.Boolean()),
            expr.BooleanFunction.Name.IsNull,
            (),
            child,
        )
        u32 = DataType(pl.UInt32())
        sum_name = next(name_generator)
        sum_agg = expr.NamedExpr(
            sum_name,
            expr.Agg(u32, "sum", (), context, expr.Cast(u32, is_null_bool)),
        )
        return [(sum_agg, True)], named_expr.reconstruct(
            expr.Cast(u32, expr.Col(u32, sum_name))
        )
    if isinstance(agg, expr.Col):
        # TODO: collect_list produces null for empty group in libcudf, empty list in polars.
        # But we need the nested value type, so need to track proper dtypes in our DSL.
        return [(named_expr, False)], named_expr.reconstruct(expr.Col(agg.dtype, name))
    if is_top and isinstance(agg, expr.Cast) and isinstance(agg.children[0], expr.Len):
        # Special case to fill nulls with zeros for empty group length calculations
        (child,) = agg.children
        child_agg, post = decompose_single_agg(
            expr.NamedExpr(next(name_generator), child),
            name_generator,
            is_top=True,
            context=context,
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
            child.dtype.plc_type
        )
        if needs_masking and agg.options:
            # pl.col("a").nan_max or nan_min
            raise NotImplementedError("Nan propagation in groupby for min/max")
        aggs, _ = decompose_single_agg(
            expr.NamedExpr(next(name_generator), child),
            name_generator,
            is_top=False,
            context=context,
        )
        if any(has_agg for _, has_agg in aggs):
            raise NotImplementedError("Nested aggs in groupby not supported")

        child_dtype = child.dtype.plc_type
        req = agg.agg_request
        is_median = agg.name == "median"
        is_quantile = agg.name == "quantile"

        # quantile agg on decimal: unsupported -> keep dtype Decimal
        # mean/median on decimal: Polars returns float -> pre-cast
        decimal_unsupported = False
        if plc.traits.is_fixed_point(child_dtype):
            if is_quantile:
                decimal_unsupported = True
            elif agg.name in {"mean", "median"}:
                tid = agg.dtype.plc_type.id()
                if tid in {plc.TypeId.FLOAT32, plc.TypeId.FLOAT64}:
                    cast_to = (
                        DataType(pl.Float64())
                        if tid == plc.TypeId.FLOAT64
                        else DataType(pl.Float32())
                    )
                    child = expr.Cast(cast_to, child)
                    child_dtype = child.dtype.plc_type

        is_group_quantile_supported = plc.traits.is_integral(
            child_dtype
        ) or plc.traits.is_floating_point(child_dtype)

        unsupported = (
            decimal_unsupported
            or ((is_median or is_quantile) and not is_group_quantile_supported)
        ) or (not plc.aggregation.is_valid_aggregation(child_dtype, req))
        if unsupported:
            return [], named_expr.reconstruct(expr.Literal(child.dtype, None))
        if needs_masking:
            child = expr.UnaryFunction(child.dtype, "mask_nans", (), child)
            # The aggregation is just reconstructed with the new
            # (potentially masked) child. This is safe because we recursed
            # to ensure there are no nested aggregations.

        # rebuild the agg with the transformed child
        new_children = [child] if not is_quantile else [child, agg.children[1]]
        named_expr = named_expr.reconstruct(agg.reconstruct(new_children))

        if agg.name == "sum":
            col = (
                expr.Cast(agg.dtype, expr.Col(DataType(pl.datatypes.Int64()), name))
                if (
                    plc.traits.is_integral(agg.dtype.plc_type)
                    and agg.dtype.id() != plc.TypeId.INT64
                )
                else expr.Col(agg.dtype, name)
            )
            # Polars semantics for sum differ by context:
            # - GROUPBY: sum(all-null group) => 0; sum(empty group) => 0  (fill-null)
            # - ROLLING: sum(all-null window) => null; sum(empty window) => 0 (fill only if empty)
            #
            # Must post-process because libcudf returns null for both empty and all-null windows/groups
            if not POLARS_VERSION_LT_1323 or context in {
                ExecutionContext.GROUPBY,
                ExecutionContext.WINDOW,
            }:
                # GROUPBY: always fill top-level nulls with 0
                return [(named_expr, True)], expr.NamedExpr(
                    name, replace_nulls(col, 0, is_top=is_top)
                )
            else:  # pragma: no cover
                # ROLLING:
                # Add a second rolling agg to compute the window size, then only
                # replace nulls with 0 when the window size is 0 (ie. empty window).
                win_len_name = next(name_generator)
                win_len = expr.NamedExpr(
                    win_len_name,
                    expr.Len(DataType(pl.Int32())),
                )

                win_len_col = expr.Col(DataType(pl.Int32()), win_len_name)
                win_len_filled = replace_nulls(win_len_col, 0, is_top=True)

                is_empty = expr.BinOp(
                    DataType(pl.Boolean()),
                    plc.binaryop.BinaryOperator.EQUAL,
                    win_len_filled,
                    expr.Literal(DataType(pl.Int32()), 0),
                )

                # If empty -> fill 0; else keep libcudf's semantics for all-null windows.
                filled = replace_nulls(col, 0, is_top=is_top)
                post_ternary_expr = expr.Ternary(agg.dtype, is_empty, filled, col)

                return [(named_expr, True), (win_len, True)], expr.NamedExpr(
                    name, post_ternary_expr
                )
        elif agg.name in {"mean", "median", "quantile", "std", "var"}:
            post_agg_col: expr.Expr = expr.Col(
                DataType(pl.Float64()), name
            )  # libcudf promotes to float64
            if agg.dtype.plc_type.id() == plc.TypeId.FLOAT32:
                # Cast back to float32 to match Polars
                post_agg_col = expr.Cast(agg.dtype, post_agg_col)
            return [(named_expr, True)], named_expr.reconstruct(post_agg_col)
        else:
            return [(named_expr, True)], named_expr.reconstruct(
                expr.Col(agg.dtype, name)
            )
    if isinstance(agg, expr.Ternary):
        when, then, otherwise = agg.children

        when_aggs, when_post = decompose_single_agg(
            expr.NamedExpr(next(name_generator), when),
            name_generator,
            is_top=False,
            context=context,
        )
        then_aggs, then_post = decompose_single_agg(
            expr.NamedExpr(next(name_generator), then),
            name_generator,
            is_top=False,
            context=context,
        )
        otherwise_aggs, otherwise_post = decompose_single_agg(
            expr.NamedExpr(next(name_generator), otherwise),
            name_generator,
            is_top=False,
            context=context,
        )

        when_has = any(h for _, h in when_aggs)
        then_has = any(h for _, h in then_aggs)
        otherwise_has = any(h for _, h in otherwise_aggs)

        if is_top and not (when_has or then_has or otherwise_has):
            raise NotImplementedError(
                "Broadcasted ternary with list output in groupby is not supported"
            )

        for post, has in (
            (when_post, when_has),
            (then_post, then_has),
            (otherwise_post, otherwise_has),
        ):
            if is_top and not has and not isinstance(post.value, expr.Literal):
                raise NotImplementedError(
                    "Broadcasting aggregated expressions in groupby/rolling"
                )

        return [*when_aggs, *then_aggs, *otherwise_aggs], named_expr.reconstruct(
            agg.reconstruct([when_post.value, then_post.value, otherwise_post.value])
        )
    if not agg.is_pointwise and isinstance(agg, expr.BooleanFunction):
        raise NotImplementedError(
            f"Non pointwise boolean function {agg.name!r} not supported in groupby or rolling context"
        )
    if agg.is_pointwise:
        aggs, posts = _decompose_aggs(
            (expr.NamedExpr(next(name_generator), child) for child in agg.children),
            name_generator,
            is_top=False,
            context=context,
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
    context: ExecutionContext,
) -> tuple[list[tuple[expr.NamedExpr, bool]], Sequence[expr.NamedExpr]]:
    new_aggs, post = zip(
        *(
            decompose_single_agg(agg, name_generator, is_top=is_top, context=context)
            for agg in aggs
        ),
        strict=True,
    )
    return list(itertools.chain.from_iterable(new_aggs)), post


def decompose_aggs(
    aggs: Iterable[expr.NamedExpr],
    name_generator: Generator[str, None, None],
    *,
    context: ExecutionContext,
) -> tuple[list[expr.NamedExpr], Sequence[expr.NamedExpr]]:
    """
    Process arbitrary aggregations into a form we can handle in grouped aggregations.

    Parameters
    ----------
    aggs
        List of aggregation expressions
    name_generator
        Generator of unique names for temporaries introduced during decomposition.
    context
        ExecutionContext in which the aggregation will run.

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
    new_aggs, post = _decompose_aggs(aggs, name_generator, is_top=True, context=context)
    return [agg for agg, _ in new_aggs], post


def apply_pre_evaluation(
    output_schema: Schema,
    keys: Sequence[expr.NamedExpr],
    original_aggs: Sequence[expr.NamedExpr],
    name_generator: Generator[str, None, None],
    context: ExecutionContext,
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
    context
        ExecutionContext in which the aggregation will run.
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
    aggs, post = decompose_aggs(original_aggs, name_generator, context=context)
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
