# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Utilities for rolling window aggregations."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pylibcudf as plc

from cudf_polars.dsl import expr, ir
from cudf_polars.dsl.expressions.base import ExecutionContext
from cudf_polars.dsl.utils.aggregations import apply_pre_evaluation
from cudf_polars.dsl.utils.naming import unique_names
from cudf_polars.dsl.utils.windows import duration_to_int

if TYPE_CHECKING:
    from collections.abc import Sequence
    from typing import Any

    from cudf_polars.typing import Schema
    from cudf_polars.utils import config

__all__ = ["rewrite_rolling"]


def rewrite_rolling(
    options: Any,
    schema: Schema,
    keys: Sequence[expr.NamedExpr],
    aggs: Sequence[expr.NamedExpr],
    config_options: config.ConfigOptions,
    inp: ir.IR,
) -> ir.IR:
    """
    Rewrite a rolling plan node into something we can handle.

    Parameters
    ----------
    options
        Rolling-specific group options.
    schema
        Schema of the rolling plan node.
    keys
        Grouping keys for the rolling node (may be empty).
    aggs
        Originally requested rolling aggregations.
    config_options
        Configuration options (currently unused).
    inp
        Input plan node to the rolling aggregation.

    Returns
    -------
    New plan node representing the rolling aggregations

    Raises
    ------
    NotImplementedError
        If any of the requested aggregations are unsupported.

    Notes
    -----
    Since libcudf can only perform rolling aggregations on columns
    (not arbitrary expressions), the approach is to split each
    aggregation into a pre-selection phase (evaluating expressions
    that live within an aggregation), the aggregation phase (now
    acting on columns only), and a post-selection phase (evaluating
    expressions of aggregated results).
    This scheme does not permit nested aggregations, so those are
    unsupported.
    """
    index_name = options.rolling.index_column
    index_dtype = schema[index_name]
    index_col = expr.Col(index_dtype, index_name)
    if (
        plc.traits.is_integral(index_dtype.plc_type)
        and index_dtype.id() != plc.TypeId.INT64
    ):
        plc_index_dtype = plc.DataType(plc.TypeId.INT64)
    else:
        plc_index_dtype = index_dtype.plc_type
    index = expr.NamedExpr(index_name, index_col)
    temp_prefix = "_" * max(map(len, schema))
    if len(aggs) > 0:
        aggs, rolling_schema, apply_post_evaluation = apply_pre_evaluation(
            schema,
            keys,
            aggs,
            unique_names(temp_prefix),
            ExecutionContext.ROLLING,
            index,
        )
    else:
        rolling_schema = schema
        apply_post_evaluation = lambda inp: inp  # noqa: E731
    preceding_ordinal = duration_to_int(plc_index_dtype, *options.rolling.offset)
    following_ordinal = duration_to_int(plc_index_dtype, *options.rolling.period)

    if (n := len(keys)) > 0:
        # Grouped rolling in polars sorts the output by the groups.
        inp = ir.Sort(
            inp.schema,
            keys,
            [plc.types.Order.ASCENDING] * n,
            [plc.types.NullOrder.BEFORE] * n,
            True,  # noqa: FBT003
            None,
            inp,
        )
    return apply_post_evaluation(
        ir.Rolling(
            rolling_schema,
            index,
            plc_index_dtype,
            preceding_ordinal,
            following_ordinal,
            options.rolling.closed_window,
            keys,
            aggs,
            options.slice,
            inp,
        )
    )
