# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
# TODO: remove need for this
# ruff: noqa: D101
"""Rolling DSL nodes."""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from functools import singledispatchmethod
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


@dataclass(frozen=True)
class UnaryOp:
    named_exprs: list[expr.NamedExpr]
    order_index: plc.Column | None = None
    by_cols_for_scan: list[Column] | None = None
    local_grouper: plc.groupby.GroupBy | None = None


@dataclass(frozen=True)
class RankOp(UnaryOp):
    pass


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


def _by_exprs(b: Expr | tuple) -> list[Expr]:
    if isinstance(b, Expr):
        return [b]
    if isinstance(b, tuple):  # pragma: no cover; tests cover this path when
        # run with the distributed scheduler only
        return [e for item in b for e in _by_exprs(item)]
    return [expr.Literal(DataType(pl.Int64()), b)]  # pragma: no cover


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

    __slots__ = (
        "_order_by_expr",
        "by_count",
        "named_aggs",
        "options",
        "post",
    )
    _non_child = (
        "dtype",
        "options",
        "named_aggs",
        "post",
        "by_count",
        "_order_by_expr",
    )

    def __init__(
        self,
        dtype: DataType,
        options: Any,
        named_aggs: Sequence[expr.NamedExpr],
        post: expr.NamedExpr,
        *by: Expr,
        _order_by_expr: Expr | None = None,
    ) -> None:
        self.dtype = dtype
        self.options = options
        self.named_aggs = tuple(named_aggs)
        self.post = post
        self.is_pointwise = False
        self._order_by_expr = _order_by_expr

        unsupported = [
            type(named_expr.value).__name__
            for named_expr in self.named_aggs
            if not (
                isinstance(named_expr.value, (expr.Len, expr.Agg))
                or (
                    isinstance(named_expr.value, expr.UnaryFunction)
                    and named_expr.value.name in {"rank"}
                )
            )
        ]
        if unsupported:
            kinds = ", ".join(sorted(set(unsupported)))
            raise NotImplementedError(
                f"Unsupported over(...) only expression: {kinds}="
            )

        # Ensures every partition-by is an Expr
        # Fixes over(1) cases with the streaming
        # executor and a small blocksize
        by_expr = [e for b in by for e in _by_exprs(b)]

        # Expose agg dependencies as children so the streaming
        # executor retains required source columns
        child_deps = [
            v.children[0]
            for ne in self.named_aggs
            for v in (ne.value,)
            if isinstance(v, expr.Agg)
            or (isinstance(v, expr.UnaryFunction) and v.name in {"rank"})
        ]
        self.by_count = len(by_expr)
        self.children = tuple(
            itertools.chain(
                by_expr,
                (() if self._order_by_expr is None else (self._order_by_expr,)),
                child_deps,
            )
        )

    def _sorted_grouper(self, by_cols_for_scan: list[Column]) -> plc.groupby.GroupBy:
        return plc.groupby.GroupBy(
            plc.Table([c.obj for c in by_cols_for_scan]),
            null_handling=plc.types.NullPolicy.INCLUDE,
            keys_are_sorted=plc.types.Sorted.YES,
            column_order=[k.order for k in by_cols_for_scan],
            null_precedence=[k.null_order for k in by_cols_for_scan],
        )

    @singledispatchmethod
    def _apply_unary_op(
        self,
        op: UnaryOp,
        _: DataFrame,
        __: plc.groupby.GroupBy,
    ) -> tuple[list[str], list[DataType], list[plc.Table]]:
        raise NotImplementedError(
            f"Unsupported unary op: {type(op).__name__}"
        )  # pragma: no cover; translation raises first

    @_apply_unary_op.register
    def _(
        self,
        op: RankOp,
        df: DataFrame,
        grouper: plc.groupby.GroupBy,
    ) -> tuple[list[str], list[DataType], list[plc.Table]]:
        rank_named = op.named_exprs
        order_index = op.order_index
        by_cols_for_scan = op.by_cols_for_scan

        rank_requests: list[plc.groupby.GroupByRequest] = []
        rank_out_names: list[str] = []
        rank_out_dtypes: list[DataType] = []

        for ne in rank_named:
            rank_expr = ne.value
            (child_expr,) = rank_expr.children
            val_col = child_expr.evaluate(df, context=ExecutionContext.FRAME).obj
            if order_index is not None:
                val_col = plc.copying.gather(
                    plc.Table([val_col]),
                    order_index,
                    plc.copying.OutOfBoundsPolicy.NULLIFY,
                ).columns()[0]
            assert isinstance(rank_expr, expr.UnaryFunction)
            method_str, descending, _ = rank_expr.options

            rank_method = {
                "average": plc.aggregation.RankMethod.AVERAGE,
                "min": plc.aggregation.RankMethod.MIN,
                "max": plc.aggregation.RankMethod.MAX,
                "dense": plc.aggregation.RankMethod.DENSE,
                "ordinal": plc.aggregation.RankMethod.FIRST,
            }[method_str]

            order = (
                plc.types.Order.DESCENDING if descending else plc.types.Order.ASCENDING
            )
            # Polars semantics: exclude nulls from domain; nulls get null ranks.
            null_precedence = (
                plc.types.NullOrder.BEFORE if descending else plc.types.NullOrder.AFTER
            )
            agg = plc.aggregation.rank(
                rank_method,
                column_order=order,
                null_handling=plc.types.NullPolicy.EXCLUDE,
                null_precedence=null_precedence,
                percentage=plc.aggregation.RankPercentage.NONE,
            )

            rank_requests.append(plc.groupby.GroupByRequest(val_col, [agg]))
            rank_out_names.append(ne.name)
            rank_out_dtypes.append(rank_expr.dtype)

        if order_index is not None and by_cols_for_scan is not None:
            # order_by expressions require us order each group
            lg = op.local_grouper
            assert isinstance(lg, plc.groupby.GroupBy)
            _, rank_tables = lg.scan(rank_requests)
        else:
            _, rank_tables = grouper.scan(rank_requests)
        return rank_out_names, rank_out_dtypes, rank_tables

    def _reorder_to_input(
        self,
        row_id: plc.Column,
        by_cols: list[Column],
        n_rows: int,
        rank_tables: list[plc.Table],
        rank_out_names: list[str],
        rank_out_dtypes: list[DataType],
        *,
        order_index: plc.Column | None = None,
    ) -> list[Column]:
        # Reorder scan results from grouped-order back to input row order
        if order_index is None:
            key_orders = [k.order for k in by_cols]
            key_nulls = [k.null_order for k in by_cols]
            order_index = plc.sorting.stable_sorted_order(
                plc.Table([*(c.obj for c in by_cols), row_id]),
                [*key_orders, plc.types.Order.ASCENDING],
                [*key_nulls, plc.types.NullOrder.AFTER],
            )

        return [
            Column(
                plc.copying.scatter(
                    plc.Table([tbl.columns()[0]]),
                    order_index,
                    plc.Table(
                        [
                            plc.Column.from_scalar(
                                plc.Scalar.from_py(None, tbl.columns()[0].type()),
                                n_rows,
                            )
                        ]
                    ),
                ).columns()[0],
                name=name,
                dtype=dtype,
            )
            for name, dtype, tbl in zip(
                rank_out_names, rank_out_dtypes, rank_tables, strict=True
            )
        ]

    def _split_named_expr(
        self,
    ) -> tuple[list[expr.NamedExpr], dict[str, list[expr.NamedExpr]]]:
        """Split into reductions vs unary window operations."""
        reductions: list[expr.NamedExpr] = []
        unary_window_ops: dict[str, list[expr.NamedExpr]] = {
            "rank": [],
        }

        for ne in self.named_aggs:
            v = ne.value
            if isinstance(v, expr.UnaryFunction) and v.name in unary_window_ops:
                unary_window_ops[v.name].append(ne)
            else:
                reductions.append(ne)
        return reductions, unary_window_ops

    def _build_window_order_index(
        self,
        by_cols: list[Column],
        *,
        row_id: plc.Column,
        order_by_col: Column | None,
        ob_desc: bool,
        ob_nulls_last: bool,
        value_col: plc.Column | None = None,
        value_desc: bool = False,
    ) -> plc.Column:
        """Compute a stable row ordering for unary operations in a grouped context."""
        cols: list[plc.Column] = [c.obj for c in by_cols]
        orders: list[plc.types.Order] = [k.order for k in by_cols]
        nulls: list[plc.types.NullOrder] = [k.null_order for k in by_cols]

        if value_col is not None:
            # for rank(...).over(...) the ranked ("sorted") order takes precedence over order_by
            cols.append(value_col)
            orders.append(
                plc.types.Order.DESCENDING if value_desc else plc.types.Order.ASCENDING
            )
            nulls.append(
                plc.types.NullOrder.BEFORE if value_desc else plc.types.NullOrder.AFTER
            )

        if order_by_col is not None:
            cols.append(order_by_col.obj)
            orders.append(
                plc.types.Order.DESCENDING if ob_desc else plc.types.Order.ASCENDING
            )
            nulls.append(
                plc.types.NullOrder.AFTER
                if ob_nulls_last
                else plc.types.NullOrder.BEFORE
            )

        # Use the row id to break ties
        cols.append(row_id)
        orders.append(plc.types.Order.ASCENDING)
        nulls.append(plc.types.NullOrder.AFTER)

        return plc.sorting.stable_sorted_order(plc.Table(cols), orders, nulls)

    def _gather_columns(
        self,
        cols: list[Column],
        order_index: plc.Column,
    ) -> list[plc.Column] | list[Column]:
        gathered_tbl = plc.copying.gather(
            plc.Table([c.obj for c in cols]),
            order_index,
            plc.copying.OutOfBoundsPolicy.NULLIFY,
        )

        return [
            Column(
                gathered_tbl.columns()[i],
                name=c.name,
                dtype=c.dtype,
                order=c.order,
                null_order=c.null_order,
                is_sorted=True,
            )
            for i, c in enumerate(cols)
        ]

    def _rank_group_by_scan(
        self,
        df: DataFrame,
        grouper: plc.groupby.GroupBy,
        rank_named: list[expr.NamedExpr],
    ) -> tuple[list[str], list[DataType], list[plc.Table]]:
        rank_requests: list[plc.groupby.GroupByRequest] = []
        rank_out_names: list[str] = []
        rank_out_dtypes: list[DataType] = []

        for ne in rank_named:
            rank_expr = ne.value
            (child_expr,) = rank_expr.children
            val_col = child_expr.evaluate(df, context=ExecutionContext.FRAME).obj
            assert isinstance(rank_expr, expr.UnaryFunction)
            method_str, descending, _ = rank_expr.options

            rank_method = {
                "average": plc.aggregation.RankMethod.AVERAGE,
                "min": plc.aggregation.RankMethod.MIN,
                "max": plc.aggregation.RankMethod.MAX,
                "dense": plc.aggregation.RankMethod.DENSE,
                "ordinal": plc.aggregation.RankMethod.FIRST,
            }[method_str]

            order = (
                plc.types.Order.DESCENDING if descending else plc.types.Order.ASCENDING
            )
            # Polars semantics: exclude nulls from domain; nulls get null ranks.
            null_precedence = (
                plc.types.NullOrder.BEFORE if descending else plc.types.NullOrder.AFTER
            )
            agg = plc.aggregation.rank(
                rank_method,
                column_order=order,
                null_handling=plc.types.NullPolicy.EXCLUDE,
                null_precedence=null_precedence,
                percentage=plc.aggregation.RankPercentage.NONE,
            )

            rank_requests.append(plc.groupby.GroupByRequest(val_col, [agg]))
            rank_out_names.append(ne.name)
            rank_out_dtypes.append(rank_expr.dtype)

        _, rank_tables = grouper.scan(rank_requests)
        return rank_out_names, rank_out_dtypes, rank_tables

    def _reorder_grouped_to_input(
        self,
        by_cols: list[Column],
        n_rows: int,
        rank_tables: list[plc.Table],
        rank_out_names: list[str],
        rank_out_dtypes: list[DataType],
    ) -> list[Column]:
        # Reorder scan results from grouped-order back to input row order
        zero = plc.Scalar.from_py(0, plc.types.SIZE_TYPE)
        one = plc.Scalar.from_py(1, plc.types.SIZE_TYPE)
        row_id = plc.filling.sequence(n_rows, zero, one)

        key_orders = [k.order for k in by_cols]
        key_nulls = [k.null_order for k in by_cols]
        grouped_order = plc.sorting.stable_sorted_order(
            plc.Table([*(c.obj for c in by_cols), row_id]),
            [*key_orders, plc.types.Order.ASCENDING],
            [*key_nulls, plc.types.NullOrder.AFTER],
        )

        return [
            Column(
                plc.copying.scatter(
                    plc.Table([tbl.columns()[0]]),
                    grouped_order,
                    plc.Table(
                        [
                            plc.Column.from_scalar(
                                plc.Scalar.from_py(None, tbl.columns()[0].type()),
                                n_rows,
                            )
                        ]
                    ),
                ).columns()[0],
                name=name,
                dtype=dtype,
            )
            for name, dtype, tbl in zip(
                rank_out_names, rank_out_dtypes, rank_tables, strict=True
            )
        ]

    def do_evaluate(  # noqa: D102
        self, df: DataFrame, *, context: ExecutionContext = ExecutionContext.FRAME
    ) -> Column:
        if context != ExecutionContext.FRAME:
            raise RuntimeError(
                "Window mapping (.over) can only be evaluated at the frame level"
            )  # pragma: no cover; translation raises first

        by_exprs = self.children[: self.by_count]
        order_by_expr = (
            self.children[self.by_count] if self._order_by_expr is not None else None
        )
        by_cols = broadcast(
            *(b.evaluate(df) for b in by_exprs),
            target_length=df.num_rows,
        )
        order_by_col = (
            broadcast(order_by_expr.evaluate(df), target_length=df.num_rows)[0]
            if order_by_expr is not None
            else None
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

        scalar_named, unary_window_ops = self._split_named_expr()

        # Build GroupByRequests for scalar aggregations
        gb_requests: list[plc.groupby.GroupByRequest] = []
        out_names: list[str] = []
        out_dtypes: list[DataType] = []
        for ne in scalar_named:
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
        # have the map between rows and groups with the correct ordering.
        left_rows = left_order.size()
        target = plc.Column.from_scalar(
            plc.Scalar.from_py(None, plc.types.SIZE_TYPE), left_rows
        )
        aligned_map = plc.copying.scatter(
            plc.Table([right_order]),
            left_order,
            plc.Table([target]),
        ).columns()[0]

        # Broadcast each scalar aggregated result back to row-shape using
        # the aligned mapping between row indices and group indices.
        broadcasted_cols = [
            Column(
                plc.copying.gather(
                    plc.Table([col]), aligned_map, plc.copying.OutOfBoundsPolicy.NULLIFY
                ).columns()[0],
                name=named_expr.name,
                dtype=dtype,
            )
            for named_expr, dtype, col in zip(
                scalar_named, out_dtypes, out_cols, strict=True
            )
        ]

        row_id = plc.filling.sequence(
            df.num_rows,
            plc.Scalar.from_py(0, plc.types.SIZE_TYPE),
            plc.Scalar.from_py(1, plc.types.SIZE_TYPE),
        )

        if rank_named := unary_window_ops["rank"]:
            if self._order_by_expr is not None:
                _, _, ob_desc, ob_nulls_last = self.options
                for ne in rank_named:
                    rank_expr = ne.value
                    assert isinstance(rank_expr, expr.UnaryFunction)
                    (child,) = rank_expr.children
                    val = child.evaluate(df, context=ExecutionContext.FRAME).obj
                    desc = rank_expr.options[1]

                    order_index = self._build_window_order_index(
                        by_cols,
                        row_id=row_id,
                        order_by_col=order_by_col,
                        ob_desc=ob_desc,
                        ob_nulls_last=ob_nulls_last,
                        value_col=val,
                        value_desc=desc,
                    )
                    by_cols_for_scan = self._gather_columns(by_cols, order_index)
                    local = self._sorted_grouper(by_cols_for_scan)
                    names, dtypes, tables = self._apply_unary_op(
                        RankOp(
                            named_exprs=[ne],
                            order_index=order_index,
                            by_cols_for_scan=by_cols_for_scan,
                            local_grouper=local,
                        ),
                        df,
                        grouper,
                    )
                    broadcasted_cols.extend(
                        self._reorder_to_input(
                            row_id,
                            by_cols,
                            df.num_rows,
                            tables,
                            names,
                            dtypes,
                            order_index=order_index,
                        )
                    )
            else:
                names, dtypes, tables = self._apply_unary_op(
                    RankOp(
                        named_exprs=rank_named, order_index=None, by_cols_for_scan=None
                    ),
                    df,
                    grouper,
                )
                broadcasted_cols.extend(
                    self._reorder_to_input(
                        row_id, by_cols, df.num_rows, tables, names, dtypes
                    )
                )

        # Create a temporary DataFrame with the broadcasted columns named by their
        # placeholder names from agg decomposition, then evaluate the post-expression.
        df = DataFrame(broadcasted_cols)
        return self.post.value.evaluate(df, context=ExecutionContext.FRAME)
