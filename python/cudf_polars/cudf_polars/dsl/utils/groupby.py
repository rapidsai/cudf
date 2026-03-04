# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Utilities for grouped aggregations."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pylibcudf as plc

from cudf_polars.dsl import ir
from cudf_polars.dsl.utils.aggregations import apply_pre_evaluation
from cudf_polars.dsl.utils.naming import unique_names

if TYPE_CHECKING:
    from collections.abc import Sequence
    from typing import Any

    from cudf_polars.dsl import expr
    from cudf_polars.utils import config

__all__ = ["rewrite_groupby"]


def rewrite_groupby(
    node: Any,
    schema: dict[str, plc.DataType],
    keys: Sequence[expr.NamedExpr],
    aggs: Sequence[expr.NamedExpr],
    config_options: config.ConfigOptions,
    inp: ir.IR,
) -> ir.IR:
    """
    Rewrite a groupby plan node into something we can handle.

    Parameters
    ----------
    node
        The polars groupby plan node.
    schema
        Schema of the groupby plan node.
    keys
        Grouping keys.
    aggs
        Originally requested aggregations.
    config_options
        Configuration options.
    inp
        Input plan node to the groupby.

    Returns
    -------
    New plan node representing the grouped aggregations.

    Raises
    ------
    NotImplementedError
        If any of the requested aggregations are unsupported.

    Notes
    -----
    Since libcudf can only perform grouped aggregations on columns
    (not arbitrary expressions), the approach is to split each
    aggregation into a pre-selection phase (evaluating expressions
    that live within an aggregation), the aggregation phase (now
    acting on columns only), and a post-selection phase (evaluating
    expressions of aggregated results).

    This does scheme does not permit nested aggregations, so those are
    unsupported.
    """
    if len(aggs) == 0:
        return ir.Distinct(
            schema,
            plc.stream_compaction.DuplicateKeepOption.KEEP_ANY,
            None,
            node.options.slice,
            node.maintain_order,
            ir.Select(schema, keys, True, inp),  # noqa: FBT003
        )

    aggs, group_schema, apply_post_evaluation = apply_pre_evaluation(
        schema, keys, aggs, unique_names(schema.keys())
    )
    # TODO: use Distinct when the partitioned executor supports it if
    # the requested aggregations are empty
    inp = ir.GroupBy(
        group_schema,
        keys,
        aggs,
        node.maintain_order,
        node.options.slice,
        config_options,
        inp,
    )
    return apply_post_evaluation(inp)
