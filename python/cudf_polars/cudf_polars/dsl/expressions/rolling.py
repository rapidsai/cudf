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
            if not (
                isinstance(named_expr.value, (expr.Len, expr.Agg))
                or (
                    isinstance(named_expr.value, expr.UnaryFunction)
                    and named_expr.value.name == "rank"
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
            or (isinstance(v, expr.UnaryFunction) and v.name == "rank")
        ]
        self.by_count = len(by_expr)
        self.children = tuple(by_expr) + tuple(child_deps)

    @staticmethod
    def _rebuild_col_with_nulls(
        ranks: plc.Column, i: plc.Column, n: int, out_dtype: DataType
    ) -> Column:
        out_plc = (
            ranks
            if ranks.type().id() == out_dtype.plc.id()
            else plc.unary.cast(ranks, out_dtype.plc)
        )
        ranks_with_nulls = plc.Column.from_scalar(
            plc.Scalar.from_py(None, out_dtype.plc), n
        )
        ranks_with_nulls = plc.copying.scatter(
            plc.Table([out_plc]), i, plc.Table([ranks_with_nulls])
        ).columns()[0]
        return Column(ranks_with_nulls, dtype=out_dtype)

    @staticmethod
    def _segmented_rank(
        values: Column,
        group_indices: plc.Column,
        *,
        method: str,
        descending: bool,
        out_dtype: DataType,
        num_groups: int,
    ) -> Column:
        """Compute per-group ranks (ordinal/dense/min/max/average)."""
        order = plc.types.Order.DESCENDING if descending else plc.types.Order.ASCENDING
        size_type = plc.types.SIZE_TYPE

        n = values.size
        zero = plc.Scalar.from_py(0, size_type)
        one = plc.Scalar.from_py(1, size_type)

        # Polars excludes nulls from ranking: so remove null rows first, but carry their
        # original indices so we can rebuild the full column (with nulls) at the end.
        i_seq = plc.filling.sequence(n, zero, one)
        full_tbl = plc.Table([values.obj, group_indices, i_seq])
        nn_tbl = plc.stream_compaction.drop_nulls(full_tbl, [0], 1)
        values, groups, i_seq = nn_tbl.columns()
        num_non_null_rows = nn_tbl.num_rows()

        # Sorting by (group, value) defines a stable permutation f : k -> i from the
        # sorted index k to the original row i. This allows us to get groups, values,
        # tied runs in sorted order.
        perm_k_to_i = plc.sorting.stable_sorted_order(
            plc.Table([groups, values]),
            [plc.types.Order.ASCENDING, order],
            [plc.types.NullOrder.AFTER, plc.types.NullOrder.AFTER],
        )

        # We also need the inverse permutation map f^{-1} : i -> k (the sorted position of
        # an original row). This lets us express ranks via the original row order.
        k_seq = plc.filling.sequence(num_non_null_rows, zero, one)
        inv_perm_i_to_k = plc.copying.scatter(
            plc.Table([k_seq]),
            perm_k_to_i,
            plc.Table([plc.Column.from_scalar(zero, num_non_null_rows)]),
        ).columns()[0]

        # Group ids in sorted order: g(f(k)).
        g_sorted = plc.copying.gather(
            plc.Table([groups]),
            perm_k_to_i,
            plc.copying.OutOfBoundsPolicy.DONT_CHECK,
        ).columns()[0]

        # For each distinct group g, calculate its first sorted position:
        # first_k(g) = min { k | g(f(k)) = g }
        first_k_per_group = plc.stream_compaction.distinct_indices(
            plc.Table([g_sorted]),
            plc.stream_compaction.DuplicateKeepOption.KEEP_FIRST,
            plc.types.NullEquality.EQUAL,
            plc.types.NanEquality.ALL_EQUAL,
        )
        group_at_first_k = plc.copying.gather(
            plc.Table([g_sorted]),
            first_k_per_group,
            plc.copying.OutOfBoundsPolicy.DONT_CHECK,
        ).columns()[0]
        first_k_at_g = plc.copying.scatter(
            plc.Table([first_k_per_group]),
            group_at_first_k,
            plc.Table([plc.Column.from_scalar(zero, num_groups)]),
        ).columns()[0]

        # Then map this to each row i by its group:
        #   first_k(i) = first_k( g(i) )
        first_k_at_i = plc.copying.gather(
            plc.Table([first_k_at_g]),
            groups,
            plc.copying.OutOfBoundsPolicy.DONT_CHECK,
        ).columns()[0]

        # Now subtract the first position of the group
        # from the globally sorted position to get
        # (0-based) ordinal rank of each row in that group.
        # (And add 1 to get the 1-based ordinal ranks)
        # Ie. ordinal(i) = f^{-1}(i) - first_k(g(i)) + 1
        ordinal_ranks = plc.binaryop.binary_operation(
            plc.binaryop.binary_operation(
                inv_perm_i_to_k,
                first_k_at_i,
                plc.binaryop.BinaryOperator.SUB,
                size_type,
            ),
            one,
            plc.binaryop.BinaryOperator.ADD,
            size_type,
        )
        if method == "ordinal":
            return GroupedRollingWindow._rebuild_col_with_nulls(
                ordinal_ranks, i_seq, n, out_dtype
            )

        # For methods dense/min/max/average we handle ties. We do this
        # by marking the start of each (group, value) run,
        # then inclusive-scan to produce run ids.
        v_sorted = plc.copying.gather(
            plc.Table([values]),
            perm_k_to_i,
            plc.copying.OutOfBoundsPolicy.DONT_CHECK,
        ).columns()[0]
        run_starts = plc.stream_compaction.distinct_indices(
            plc.Table([g_sorted, v_sorted]),
            plc.stream_compaction.DuplicateKeepOption.KEEP_FIRST,
            plc.types.NullEquality.EQUAL,
            plc.types.NanEquality.ALL_EQUAL,
        )
        num_tied_runs = run_starts.size()
        run_markers = plc.copying.scatter(
            plc.Table([plc.Column.from_scalar(one, num_tied_runs)]),
            run_starts,
            plc.Table([plc.Column.from_scalar(zero, num_non_null_rows)]),
        ).columns()[0]
        run_id_sorted = plc.reduce.scan(
            run_markers, plc.aggregation.sum(), plc.reduce.ScanType.INCLUSIVE
        )
        run_id = plc.copying.gather(
            plc.Table([run_id_sorted]),
            inv_perm_i_to_k,
            plc.copying.OutOfBoundsPolicy.DONT_CHECK,
        ).columns()[0]

        if method == "dense":
            # Dense rank assigns consecutive integers to distinct values within each group.
            # dense(i) = run_id(i) - run_id_first_k_per_row(i) + 1
            # run_id_first_k_per_row(i) = the run id at the group's first sorted row
            run_id_at_first_k = plc.copying.gather(
                plc.Table([run_id_sorted]),
                first_k_per_group,
                plc.copying.OutOfBoundsPolicy.DONT_CHECK,
            ).columns()[0]
            run_id_first_k_map = plc.copying.scatter(
                plc.Table([run_id_at_first_k]),
                group_at_first_k,
                plc.Table([plc.Column.from_scalar(zero, num_groups)]),
            ).columns()[0]
            run_id_first_k_per_row = plc.copying.gather(
                plc.Table([run_id_first_k_map]),
                groups,
                plc.copying.OutOfBoundsPolicy.DONT_CHECK,
            ).columns()[0]
            dense_ranks = plc.binaryop.binary_operation(
                plc.binaryop.binary_operation(
                    run_id,
                    run_id_first_k_per_row,
                    plc.binaryop.BinaryOperator.SUB,
                    size_type,
                ),
                one,
                plc.binaryop.BinaryOperator.ADD,
                size_type,
            )
            return GroupedRollingWindow._rebuild_col_with_nulls(
                dense_ranks, i_seq, n, out_dtype
            )

        # For min/max/average we work with per-group positions in the sorted view.
        # For each sorted row k, its 0-based position within the group is:
        #   pos_in_group(k) = k - first_k(g(f(k))).
        first_sorted_k_for_group = plc.copying.gather(
            plc.Table([first_k_at_g]),
            g_sorted,
            plc.copying.OutOfBoundsPolicy.DONT_CHECK,
        ).columns()[0]
        pos_in_group = plc.binaryop.binary_operation(
            k_seq,
            first_sorted_k_for_group,
            plc.binaryop.BinaryOperator.SUB,
            size_type,
        )

        # Each tie run has a min and max position within its group.
        # We compute run_ends from run_starts, then gather min/max positions
        # per run, and finally convert to 1-based ranks.
        run_ends = plc.Column.from_scalar(
            plc.Scalar.from_py(num_non_null_rows - 1, size_type), num_tied_runs
        )
        if num_tied_runs > 1:
            # There are multiple tied runs, so update the default run_ends
            # for each run run_id (r) except the last, set run_ends[r] = run_starts[r+1] - 1.
            # This ensures that each run's end index is the position immediately
            # before the start of the next run. The last run already ends at
            # num_non_null_rows - 1.
            tail = plc.copying.gather(
                plc.Table([run_starts]),
                plc.filling.sequence(num_tied_runs - 1, one, one),
                plc.copying.OutOfBoundsPolicy.DONT_CHECK,
            ).columns()[0]
            tail = plc.binaryop.binary_operation(
                tail, one, plc.binaryop.BinaryOperator.SUB, size_type
            )
            run_ends = plc.copying.scatter(
                plc.Table([tail]),
                plc.filling.sequence(num_tied_runs - 1, zero, one),
                plc.Table([run_ends]),
            ).columns()[0]

        # For each run (r), get the min and max position within the group
        # from pos_in_group at run_starts[r] and run_ends[r].
        min_pos = plc.copying.gather(
            plc.Table([pos_in_group]),
            run_starts,
            plc.copying.OutOfBoundsPolicy.DONT_CHECK,
        ).columns()[0]
        max_pos = plc.copying.gather(
            plc.Table([pos_in_group]),
            run_ends,
            plc.copying.OutOfBoundsPolicy.DONT_CHECK,
        ).columns()[0]
        min_pos = plc.binaryop.binary_operation(
            min_pos, one, plc.binaryop.BinaryOperator.ADD, size_type
        )
        max_pos = plc.binaryop.binary_operation(
            max_pos, one, plc.binaryop.BinaryOperator.ADD, size_type
        )

        def _per_row_ranks_from_run_pos(positions: plc.Column) -> plc.Column:
            # Map the per-run min/max ranks back to per-row results
            run_ids_at_starts = plc.copying.gather(
                plc.Table([run_id_sorted]),
                run_starts,
                plc.copying.OutOfBoundsPolicy.DONT_CHECK,
            ).columns()[0]
            run_index_per_row = plc.binaryop.binary_operation(
                run_id_sorted, one, plc.binaryop.BinaryOperator.SUB, size_type
            )
            run_index_per_run = plc.binaryop.binary_operation(
                run_ids_at_starts, one, plc.binaryop.BinaryOperator.SUB, size_type
            )

            mapping = plc.copying.scatter(
                plc.Table([positions]),
                run_index_per_run,
                plc.Table(
                    [
                        plc.Column.from_scalar(
                            plc.Scalar.from_py(0, size_type), num_tied_runs
                        )
                    ]
                ),
            ).columns()[0]

            ranks_sorted = plc.copying.gather(
                plc.Table([mapping]),
                run_index_per_row,
                plc.copying.OutOfBoundsPolicy.DONT_CHECK,
            ).columns()[0]

            return plc.copying.gather(
                plc.Table([ranks_sorted]),
                inv_perm_i_to_k,
                plc.copying.OutOfBoundsPolicy.DONT_CHECK,
            ).columns()[0]

        if method == "min":
            return GroupedRollingWindow._rebuild_col_with_nulls(
                _per_row_ranks_from_run_pos(min_pos), i_seq, n, out_dtype
            )
        if method == "max":
            return GroupedRollingWindow._rebuild_col_with_nulls(
                _per_row_ranks_from_run_pos(max_pos), i_seq, n, out_dtype
            )
        if method == "average":
            max_ranks = _per_row_ranks_from_run_pos(max_pos)
            min_ranks = _per_row_ranks_from_run_pos(min_pos)
            f64 = plc.DataType(plc.TypeId.FLOAT64)
            min_ = plc.unary.cast(min_ranks, f64)
            max_ = plc.unary.cast(max_ranks, f64)
            sum_ = plc.binaryop.binary_operation(
                min_, max_, plc.binaryop.BinaryOperator.ADD, f64
            )
            two = plc.Scalar.from_py(2.0, f64)
            avg = plc.binaryop.binary_operation(
                sum_, two, plc.binaryop.BinaryOperator.DIV, f64
            )
            if out_dtype.plc.id() != f64.id():  # pragma: no cover
                avg = plc.unary.cast(avg, out_dtype.plc)
            return GroupedRollingWindow._rebuild_col_with_nulls(
                avg, i_seq, n, out_dtype
            )

        raise NotImplementedError(f"rank({method=}).over(..)")  # pragma: no cover

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

        # Split up expressions into scalar aggs (eg. Len) vs per-row (eg. rank)
        scalar_named: list[expr.NamedExpr] = []
        rank_named: list[expr.NamedExpr] = []
        for ne in self.named_aggs:
            v = ne.value
            if isinstance(v, expr.UnaryFunction) and v.name == "rank":
                rank_named.append(ne)
            else:
                scalar_named.append(ne)

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

        # Broadcast each scalar aggregated result back to row-shape using
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
                scalar_named, out_dtypes, out_cols, strict=True
            )
        ]

        # Compute per-row ranks over groups using the aligned group ids
        if rank_named:
            group_indices = aligned_map
            num_groups = group_keys_tbl.num_rows()

            for ne in rank_named:
                rank_expr = ne.value
                (child_expr,) = rank_expr.children
                values = child_expr.evaluate(df, context=ExecutionContext.FRAME)
                assert isinstance(rank_expr, expr.UnaryFunction)
                method_str, descending, _ = rank_expr.options
                ranked = GroupedRollingWindow._segmented_rank(
                    values,
                    group_indices,
                    method=method_str,
                    descending=bool(descending),
                    out_dtype=rank_expr.dtype,
                    num_groups=num_groups,
                )
                ranked.name = ne.name
                broadcasted_cols.append(ranked)

        # Create a temporary DataFrame with the broadcasted columns named by their
        # placeholder names from agg decomposition, then evaluate the post-expression.
        df = DataFrame(broadcasted_cols)
        return self.post.value.evaluate(df, context=ExecutionContext.FRAME)
