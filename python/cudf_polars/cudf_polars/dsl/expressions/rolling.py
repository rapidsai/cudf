# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
# TODO: remove need for this
# ruff: noqa: D101
"""Rolling DSL nodes."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import polars as pl

import pylibcudf as plc

from cudf_polars.containers import Column, DataFrame, DataType
from cudf_polars.dsl import expr
from cudf_polars.dsl.expressions.base import ExecutionContext, Expr
from cudf_polars.dsl.utils.reshape import broadcast
from cudf_polars.dsl.utils.windows import offsets_to_windows, range_window_bounds

if TYPE_CHECKING:
    from collections.abc import Sequence

    from cudf_polars.typing import ClosedInterval, Duration

__all__ = ["GroupedRollingWindow", "RollingWindow", "to_request"]


def to_request(
    value: expr.Expr, orderby: Column, df: DataFrame
) -> plc.rolling.RollingRequest:
    """
    Produce a rolling request for evaluation with pylibcudf.

    Parameters
    ----------
    value
        The expression to perform the rolling aggregation on.
    orderby
        Orderby column, used as input to the request when the aggregation is Len.
    df
        DataFrame used to evaluate the inputs to the aggregation.
    """
    min_periods = 1
    if isinstance(value, expr.Len):
        # A count aggregation, we need a column so use the orderby column
        col = orderby
    elif isinstance(value, expr.Agg):
        child = value.children[0]
        col = child.evaluate(df, context=ExecutionContext.ROLLING)
        if value.name == "var":
            # Polars variance produces null if nvalues <= ddof
            # libcudf produces NaN. However, we can get the polars
            # behaviour by setting the minimum window size to ddof +
            # 1.
            min_periods = value.options + 1
    else:
        col = value.evaluate(
            df, context=ExecutionContext.ROLLING
        )  # pragma: no cover; raise before we get here because we
        # don't do correct handling of empty groups
    return plc.rolling.RollingRequest(col.obj, min_periods, value.agg_request)


class RollingWindow(Expr):
    __slots__ = (
        "closed_window",
        "following",
        "offset",
        "orderby",
        "orderby_dtype",
        "period",
        "preceding",
    )
    _non_child = (
        "dtype",
        "orderby_dtype",
        "offset",
        "period",
        "closed_window",
        "orderby",
    )

    def __init__(
        self,
        dtype: DataType,
        orderby_dtype: DataType,
        offset: Duration,
        period: Duration,
        closed_window: ClosedInterval,
        orderby: str,
        agg: Expr,
    ) -> None:
        self.dtype = dtype
        self.orderby_dtype = orderby_dtype
        # NOTE: Save original `offset` and `period` args,
        # because the `preceding` and `following` attributes
        # cannot be serialized (and must be reconstructed
        # within `__init__`).
        self.offset = offset
        self.period = period
        self.preceding, self.following = offsets_to_windows(
            orderby_dtype, offset, period
        )
        self.closed_window = closed_window
        self.orderby = orderby
        self.children = (agg,)
        self.is_pointwise = False
        if agg.agg_request.kind() == plc.aggregation.Kind.COLLECT_LIST:
            raise NotImplementedError(
                "Incorrect handling of empty groups for list collection"
            )
        if not plc.rolling.is_valid_rolling_aggregation(agg.dtype.plc, agg.agg_request):
            raise NotImplementedError(f"Unsupported rolling aggregation {agg}")

    def do_evaluate(  # noqa: D102
        self, df: DataFrame, *, context: ExecutionContext = ExecutionContext.FRAME
    ) -> Column:
        if context != ExecutionContext.FRAME:
            raise RuntimeError(
                "Rolling aggregation inside groupby/over/rolling"
            )  # pragma: no cover; translation raises first
        (agg,) = self.children
        orderby = df.column_map[self.orderby]
        # Polars casts integral orderby to int64, but only for calculating window bounds
        if (
            plc.traits.is_integral(orderby.obj.type())
            and orderby.obj.type().id() != plc.TypeId.INT64
        ):
            orderby_obj = plc.unary.cast(orderby.obj, plc.DataType(plc.TypeId.INT64))
        else:
            orderby_obj = orderby.obj
        preceding, following = range_window_bounds(
            self.preceding, self.following, self.closed_window
        )
        if orderby.obj.null_count() != 0:
            raise RuntimeError(
                f"Index column '{self.orderby}' in rolling may not contain nulls"
            )
        if not orderby.check_sorted(
            order=plc.types.Order.ASCENDING, null_order=plc.types.NullOrder.BEFORE
        ):
            raise RuntimeError(
                f"Index column '{self.orderby}' in rolling is not sorted, please sort first"
            )
        (result,) = plc.rolling.grouped_range_rolling_window(
            plc.Table([]),
            orderby_obj,
            plc.types.Order.ASCENDING,
            plc.types.NullOrder.BEFORE,
            preceding,
            following,
            [to_request(agg, orderby, df)],
        ).columns()
        return Column(result, dtype=self.dtype)


class GroupedRollingWindow(Expr):
    """
    Compute a window ``.over(...)`` aggregation and broadcast to rows.

    Notes
    -----
    - This expression node currently implements **grouped window mapping**
      (aggregate once per group, then broadcast back), not rolling windows.
    - It can be extended later to support `rolling(...).over(...)`
      when polars supports that expression.
    """

    __slots__ = ("by_count", "named_aggs", "options", "post")
    _non_child = ("dtype", "options", "named_aggs", "post", "by_count")

    def __init__(
        self,
        dtype: DataType,
        options: Any,
        named_aggs: Sequence[expr.NamedExpr],
        post: expr.NamedExpr,
        *by: Expr,
    ) -> None:
        self.dtype = dtype
        self.options = options
        self.named_aggs = tuple(named_aggs)
        self.post = post
        self.is_pointwise = False

        unsupported = [
            type(named_expr.value).__name__
            for named_expr in self.named_aggs
            if not isinstance(named_expr.value, (expr.Len, expr.Agg))
        ]
        if unsupported:
            kinds = ", ".join(sorted(set(unsupported)))
            raise NotImplementedError(
                f"Unsupported over(...) only expression: {kinds}="
            )

        # Ensures every partition-by is an Expr
        # Fixes over(1) cases with the streaming
        # executor and a small blocksize
        by_expr = [
            (b if isinstance(b, Expr) else expr.Literal(DataType(pl.Int64()), b))
            for b in by
        ]

        # Expose agg dependencies as children so the streaming
        # executor retains required source columns
        child_deps = [
            v.children[0]
            for ne in self.named_aggs
            for v in (ne.value,)
            if isinstance(v, expr.Agg)
        ]
        self.by_count = len(by_expr)
        self.children = tuple(by_expr) + tuple(child_deps)

    def do_evaluate(  # noqa: D102
        self, df: DataFrame, *, context: ExecutionContext = ExecutionContext.FRAME
    ) -> Column:
        if context != ExecutionContext.FRAME:
            raise RuntimeError(
                "Window mapping (.over) can only be evaluated at the frame level"
            )  # pragma: no cover; translation raises first

        by_exprs = self.children[: self.by_count]
        by_cols = broadcast(
            *(b.evaluate(df) for b in by_exprs),
            target_length=df.num_rows,
        )

        by_tbl = plc.Table([c.obj for c in by_cols])

        sorted_flag = (
            plc.types.Sorted.YES
            if all(k.is_sorted for k in by_cols)
            else plc.types.Sorted.NO
        )
        grouper = plc.groupby.GroupBy(
            by_tbl,
            null_handling=plc.types.NullPolicy.INCLUDE,
            keys_are_sorted=sorted_flag,
            column_order=[k.order for k in by_cols],
            null_precedence=[k.null_order for k in by_cols],
        )

        gb_requests: list[plc.groupby.GroupByRequest] = []
        out_names: list[str] = []
        out_dtypes: list[DataType] = []
        for ne in self.named_aggs:
            val = ne.value
            out_names.append(ne.name)
            out_dtypes.append(val.dtype)

            if isinstance(val, expr.Len):
                # A count aggregation, we need a column so use a key column
                col = by_cols[0].obj
                gb_requests.append(plc.groupby.GroupByRequest(col, [val.agg_request]))
            elif isinstance(val, expr.Agg):
                (child,) = (
                    val.children if val.name != "quantile" else (val.children[0],)
                )
                col = child.evaluate(df, context=ExecutionContext.FRAME).obj
                gb_requests.append(plc.groupby.GroupByRequest(col, [val.agg_request]))

        group_keys_tbl, value_tables = grouper.aggregate(gb_requests)
        out_cols = (t.columns()[0] for t in value_tables)

        # We do a left-join between the input keys to group-keys
        # so every input row appears exactly once. left_order is
        # returned un-ordered by libcudf.
        left_order, right_order = plc.join.left_join(
            by_tbl, group_keys_tbl, plc.types.NullEquality.EQUAL
        )

        # Scatter the right order indices into an all-null table
        # and at the position of the index in left order. Now we
        # the map between rows an groups with the correct ordering
        left_rows = left_order.size()
        target = plc.Column.from_scalar(
            plc.Scalar.from_py(None, plc.types.SIZE_TYPE), left_rows
        )
        aligned_map = plc.copying.scatter(
            plc.Table([right_order]),
            left_order,
            plc.Table([target]),
        ).columns()[0]

        # Broadcast each aggregated result back to row-shape using
        # the aligned mapping between rows indices and group indices
        broadcasted_cols = [
            Column(
                plc.copying.gather(
                    plc.Table([col]), aligned_map, plc.copying.OutOfBoundsPolicy.NULLIFY
                ).columns()[0],
                name=named_expr.name,
                dtype=dtype,
            )
            for named_expr, dtype, col in zip(
                self.named_aggs, out_dtypes, out_cols, strict=True
            )
        ]

        # Create a temporary DataFrame with the broadcasted columns named by their
        # placeholder names from agg decomposition, then evaluate the post-expression.
        df = DataFrame(broadcasted_cols)
        return self.post.value.evaluate(df, context=ExecutionContext.FRAME)
