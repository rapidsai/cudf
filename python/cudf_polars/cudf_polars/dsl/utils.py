# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Utilities for DSL creation."""

from __future__ import annotations

import itertools
from typing import TYPE_CHECKING

import pylibcudf as plc

from cudf_polars.dsl import expr

if TYPE_CHECKING:
    from collections.abc import Generator, Iterable, Sequence


__all__ = ["decompose_aggs", "unique_names"]


def unique_names(prefix: str) -> Generator[str, None, None]:
    """
    Generate unique names with a given prefix.

    Parameters
    ----------
    prefix
        Prefix to give to names

    Notes
    -----
    If creating temporary named expressions for a node, create a
    prefix that is as long as the longest key in the schema (doesn't
    matter what it is) and use that.

    Yields
    ------
    Unique names (just using sequence numbers)
    """
    i = 0
    while True:
        yield f"{prefix}{i}"
        i += 1


def decompose_single_agg(
    named_expr: expr.NamedExpr, name_generator: Generator[str, None, None]
) -> tuple[list[expr.NamedExpr], list[expr.NamedExpr], expr.NamedExpr, bool]:
    """
    Decompose a single named aggregation.

    Parameters
    ----------
    named_expr
        The named aggregation to decompose
    name_generator
        Generator of unique names for temporaries introduced during decomposition.

    Returns
    -------
    tuple
        Four-tuple of list of expressions to apply in a pre-processing
        phase, list of expressions to apply as grouped aggregations,
        single expression to apply to post-process the grouped
        aggregations, and a boolean indicating whether processing in
        the inner expression requires aggregations.

    Raises
    ------
    NotImplementedError
        If the expression contains nested aggregations or unsupported
        operations in a grouped aggregation context.
    """
    agg = named_expr.value
    name = named_expr.name
    if isinstance(agg, expr.Col):
        return [named_expr], [named_expr], named_expr, False
    if isinstance(agg, expr.Len):
        return (
            [named_expr],
            [named_expr],
            expr.NamedExpr(name, expr.Col(agg.dtype, name)),
            False,
        )
    if isinstance(agg, (expr.Literal, expr.LiteralColumn)):
        return [named_expr], [], named_expr, False
    if isinstance(agg, expr.Agg):
        (child,) = agg.children
        needs_masking = agg.name in {"min", "max"} and plc.traits.is_floating_point(
            child.dtype
        )
        if needs_masking and agg.options:
            # pl.col("a").nan_max or nan_min
            raise NotImplementedError("Nan propagation in groupby for min/max")
        pre, _, post, has_agg = decompose_single_agg(
            expr.NamedExpr(next(name_generator), child), name_generator
        )
        if has_agg:
            raise NotImplementedError("Nested aggs in groupby not supported")
        if needs_masking:
            (obj,) = pre
            pre = [
                expr.NamedExpr(
                    obj.name,
                    expr.UnaryFunction(obj.value.dtype, "mask_nans", (), obj.value),
                )
            ]
        # We apply the aggregation to the child "after"
        # post-processing. We know this is safe because there are no
        # nested aggregations.
        return (
            pre,
            [expr.NamedExpr(name, agg.reconstruct([post.value]))],
            expr.NamedExpr(name, expr.Col(agg.dtype, name)),
            True,
        )
    if isinstance(agg, expr.Cast):
        (child,) = agg.children
        pre, group, post, has_agg = decompose_single_agg(
            expr.NamedExpr(next(name_generator), child), name_generator
        )
        return (
            pre,
            group,
            expr.NamedExpr(name, agg.reconstruct([post.value])),
            has_agg,
        )
    if isinstance(agg, expr.UnaryFunction):
        if not agg.is_pointwise:
            raise NotImplementedError("Non-pointwise unary function inside groupby")
        (child,) = agg.children
        pre, group, post, has_agg = decompose_single_agg(
            expr.NamedExpr(next(name_generator), child), name_generator
        )
        if has_agg:
            return (
                pre,
                group,
                expr.NamedExpr(name, agg.reconstruct([post.value])),
                has_agg,
            )
        else:
            col = expr.NamedExpr(name, expr.Col(agg.dtype, name))
            return [named_expr], [col], col, False
    if isinstance(agg, (expr.BinOp, expr.StringFunction, expr.TemporalFunction)):
        pres, groups, posts, has_aggs = _decompose_aggs(
            (expr.NamedExpr(next(name_generator), child) for child in agg.children),
            name_generator,
        )
        if any(has_aggs):
            return (
                pres,
                groups,
                expr.NamedExpr(name, agg.reconstruct([p.value for p in posts])),
                True,
            )
        else:
            col = expr.NamedExpr(name, expr.Col(agg.dtype, name))
            return [named_expr], [col], col, False
    raise NotImplementedError(f"No support for {type(agg)} in groupby")


def _decompose_aggs(
    aggs: Iterable[expr.NamedExpr], name_generator: Generator[str, None, None]
) -> tuple[
    list[expr.NamedExpr], list[expr.NamedExpr], Sequence[expr.NamedExpr], Sequence[bool]
]:
    pre, group, post, has_agg = zip(
        *(decompose_single_agg(agg, name_generator) for agg in aggs), strict=True
    )
    return (
        list(itertools.chain.from_iterable(pre)),
        list(itertools.chain.from_iterable(group)),
        post,
        has_agg,
    )


def decompose_aggs(
    aggs: Iterable[expr.NamedExpr], name_generator: Generator[str, None, None]
) -> tuple[list[expr.NamedExpr], list[expr.NamedExpr], Sequence[expr.NamedExpr]]:
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
    tuple
        Three-tuple of expressions to evaluate before the aggregation
        (as a ``Select`), aggregation expressions for the ``GroupBy``
        node, and expressions to evaluate after the aggregation (as a
        ``Select``).
    """
    pre, group, post, _ = _decompose_aggs(aggs, name_generator)
    return pre, group, post
