# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
# TODO: remove need for this
# ruff: noqa: D101
"""
DSL nodes for the LogicalPlan of polars.

An IR node is either a source, normal, or a sink. Respectively they
can be considered as functions:

- source: `IO () -> DataFrame`
- normal: `DataFrame -> DataFrame`
- sink: `DataFrame -> IO ()`
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cache
from typing import TYPE_CHECKING, Any, Callable, ClassVar

import pyarrow as pa
from typing_extensions import assert_never

import polars as pl

import cudf
import cudf._lib.pylibcudf as plc

import cudf_polars.dsl.expr as expr
from cudf_polars.containers import Column, DataFrame
from cudf_polars.utils import dtypes, sorting

if TYPE_CHECKING:
    from typing import Literal

    from cudf_polars.dsl.expr import Expr


__all__ = [
    "IR",
    "PythonScan",
    "Scan",
    "Cache",
    "DataFrameScan",
    "Select",
    "GroupBy",
    "Join",
    "HStack",
    "Distinct",
    "Sort",
    "Slice",
    "Filter",
    "Projection",
    "MapFunction",
    "Union",
    "HConcat",
    "ExtContext",
]


@dataclass(slots=True)
class IR:
    schema: dict

    def evaluate(self, *, cache: dict[int, DataFrame]) -> DataFrame:
        """Evaluate and return a dataframe."""
        raise NotImplementedError


@dataclass(slots=True)
class PythonScan(IR):
    options: Any
    predicate: Expr | None


@dataclass(slots=True)
class Scan(IR):
    typ: Any
    paths: list[str]
    file_options: Any
    predicate: Expr | None

    def __post_init__(self):
        """Validate preconditions."""
        if self.file_options.n_rows is not None:
            raise NotImplementedError("row limit in scan")
        if self.typ not in ("csv", "parquet"):
            raise NotImplementedError(f"Unhandled scan type: {self.typ}")

    def evaluate(self, *, cache: dict[int, DataFrame]) -> DataFrame:
        """Evaluate and return a dataframe."""
        options = self.file_options
        n_rows = options.n_rows
        with_columns = options.with_columns
        row_index = options.row_index
        assert n_rows is None
        if self.typ == "csv":
            df = DataFrame.from_cudf(
                cudf.concat(
                    [cudf.read_csv(p, usecols=with_columns) for p in self.paths]
                )
            )
        elif self.typ == "parquet":
            df = DataFrame.from_cudf(
                cudf.read_parquet(self.paths, columns=with_columns)
            )
        else:
            assert_never(self.typ)
        if row_index is not None:
            name, offset = row_index
            dtype = dtypes.from_polars(self.schema[name])
            step = plc.interop.from_arrow(pa.scalar(1), data_type=dtype)
            init = plc.interop.from_arrow(pa.scalar(offset), data_type=dtype)
            index = Column(
                plc.filling.sequence(df.num_rows(), init, step), name
            ).set_sorted(
                is_sorted=plc.types.Sorted.YES,
                order=plc.types.Order.ASCENDING,
                null_order=plc.types.null_order.AFTER,
            )
            df = df.with_columns([index])
        if self.predicate is None:
            return df
        else:
            mask = self.predicate.evaluate(df)
            return df.filter(mask)


@dataclass(slots=True)
class Cache(IR):
    key: int
    value: IR

    def evaluate(self, *, cache: dict[int, DataFrame]) -> DataFrame:
        """Evaluate and return a dataframe."""
        try:
            return cache[self.key]
        except KeyError:
            return cache.setdefault(self.key, self.value.evaluate(cache=cache))


@dataclass(slots=True)
class DataFrameScan(IR):
    df: Any
    projection: list[str]
    predicate: Expr | None

    def evaluate(self, *, cache: dict[int, DataFrame]) -> DataFrame:
        """Evaluate and return a dataframe."""
        pdf = pl.DataFrame._from_pydf(self.df)
        if self.projection is not None:
            pdf = pdf.select(self.projection)
        # TODO: goes away when libcudf supports large strings
        table = pdf.to_arrow()
        schema = table.schema
        for i, field in enumerate(schema):
            if field.type == pa.large_string():
                # TODO: Nested types
                schema = schema.set(i, pa.field(field.name, pa.string()))
        table = table.cast(schema)
        df = DataFrame.from_table(
            plc.interop.from_arrow(table), list(self.schema.keys())
        )
        if self.predicate is not None:
            mask = self.predicate.evaluate(df)
            return df.filter(mask)
        else:
            return df


@dataclass(slots=True)
class Select(IR):
    df: IR
    cse: list[Expr]
    expr: list[Expr]

    def evaluate(self, *, cache: dict[int, DataFrame]):
        """Evaluate and return a dataframe."""
        df = self.df.evaluate(cache=cache)
        for e in self.cse:
            df = df.with_columns([e.evaluate(df)])
        return DataFrame([e.evaluate(df) for e in self.expr], [])


@dataclass(slots=True)
class GroupBy(IR):
    df: IR
    agg_requests: list[Expr]
    keys: list[Expr]
    options: Any

    @staticmethod
    def check_agg(agg: Expr) -> int:
        """
        Determine if we can handle an aggregation expression.

        Parameters
        ----------
        agg
            Expression to check

        Returns
        -------
        depth of nesting

        Raises
        ------
        NotImplementedError for unsupported expression nodes.
        """
        if isinstance(agg, expr.Agg):
            if agg.name == "implode":
                raise NotImplementedError("implode in groupby")
            return 1 + GroupBy.check_agg(agg.column)
        elif isinstance(agg, (expr.Len, expr.Column, expr.Literal)):
            return 0
        elif isinstance(agg, expr.BinOp):
            return max(GroupBy.check_agg(agg.left), GroupBy.check_agg(agg.right))
        elif isinstance(agg, expr.Cast):
            return GroupBy.check_agg(agg.column)
        else:
            raise NotImplementedError(f"No handler for {agg=}")

    def __post_init__(self):
        """Check whether all the aggregations are implemented."""
        if any(GroupBy.check_agg(a) > 1 for a in self.agg_requests):
            raise NotImplementedError("Nested aggregations in groupby")


@dataclass(slots=True)
class Join(IR):
    left: IR
    right: IR
    left_on: list[Expr]
    right_on: list[Expr]
    options: Any

    def __post_init__(self):
        """Validate preconditions."""
        if self.options[0] == "cross":
            raise NotImplementedError("cross join not implemented")

    @cache
    @staticmethod
    def _joiners(
        how: Literal["inner", "left", "outer", "leftsemi", "leftanti"],
    ) -> tuple[
        Callable, plc.copying.OutOfBoundsPolicy, plc.copying.OutOfBoundsPolicy | None
    ]:
        if how == "inner":
            return (
                plc.join.inner_join,
                plc.copying.OutOfBoundsPolicy.DONT_CHECK,
                plc.copying.OutOfBoundsPolicy.DONT_CHECK,
            )
        elif how == "left":
            return (
                plc.join.left_join,
                plc.copying.OutOfBoundsPolicy.DONT_CHECK,
                plc.copying.OutOfBoundsPolicy.NULLIFY,
            )
        elif how == "outer":
            return (
                plc.join.full_join,
                plc.copying.OutOfBoundsPolicy.NULLIFY,
                plc.copying.OutOfBoundsPolicy.NULLIFY,
            )
        elif how == "leftsemi":
            return (
                plc.join.left_semi_join,
                plc.copying.OutOfBoundsPolicy.DONT_CHECK,
                None,
            )
        elif how == "leftanti":
            return (
                plc.join.left_anti_join,
                plc.copying.OutOfBoundsPolicy.DONT_CHECK,
                None,
            )
        else:
            assert_never(how)

    def evaluate(self, *, cache: dict[int, DataFrame]) -> DataFrame:
        """Evaluate and return a dataframe."""
        left = self.left.evaluate(cache=cache)
        right = self.right.evaluate(cache=cache)
        left_on = DataFrame([e.evaluate(left) for e in self.left_on], [])
        right_on = DataFrame([e.evaluate(right) for e in self.right_on], [])
        how, join_nulls, zlice, suffix, coalesce = self.options
        null_equality = (
            plc.types.NullEquality.EQUAL
            if join_nulls
            else plc.types.NullEquality.UNEQUAL
        )
        suffix = "_right" if suffix is None else suffix
        join_fn, left_policy, right_policy = Join._joiners(how)
        if right_policy is None:
            # Semi join
            lg = join_fn(left_on.table, right_on.table, null_equality)
            left = left.replace_columns(*left_on.columns)
            table = plc.copying.gather(left.table, lg, left_policy)
            result = DataFrame.from_table(table, left.column_names)
        else:
            lg, rg = join_fn(left_on, right_on, null_equality)
            left = left.replace_columns(*left_on.columns)
            right = right.replace_columns(*right_on.columns)
            if coalesce and how != "outer":
                right = right.discard_columns(set(right_on.names))
            left = DataFrame.from_table(
                plc.copying.gather(left.table, lg, left_policy), left.column_names
            )
            right = DataFrame.from_table(
                plc.copying.gather(right.table, rg, right_policy), right.column_names
            )
            if coalesce and how == "outer":
                left.replace_columns(
                    *(
                        Column(
                            plc.replace.replace_nulls(left_col.obj, right_col.obj),
                            left_col.name,
                        )
                        for left_col, right_col in zip(
                            left.select_columns(set(left_on.names)),
                            right.select_columns(set(right_on.names)),
                        )
                    )
                )
                right.discard_columns(set(right_on.names))
            right = right.rename_columns(
                {name: f"{name}{suffix}" for name in right.names if name in left.names}
            )
            result = left.with_columns(right.columns)
        return result.slice(zlice)


@dataclass(slots=True)
class HStack(IR):
    df: IR
    columns: list[Expr]

    def evaluate(self, *, cache: dict[int, DataFrame]) -> DataFrame:
        """Evaluate and return a dataframe."""
        df = self.df.evaluate(cache=cache)
        return df.with_columns([c.evaluate(df) for c in self.columns])


@dataclass(slots=True)
class Distinct(IR):
    df: IR
    keep: plc.stream_compaction.DuplicateKeepOption
    subset: set[str] | None
    zlice: tuple[int, int] | None
    stable: bool

    _KEEP_MAP: ClassVar[dict[str, plc.stream_compaction.DuplicateKeepOption]] = {
        "first": plc.stream_compaction.DuplicateKeepOption.KEEP_FIRST,
        "last": plc.stream_compaction.DuplicateKeepOption.KEEP_LAST,
        "none": plc.stream_compaction.DuplicateKeepOption.KEEP_NONE,
        "any": plc.stream_compaction.DuplicateKeepOption.KEEP_ANY,
    }

    def __init__(self, schema: dict, df: IR, options: Any):
        self.schema = schema
        self.df = df
        (keep, subset, maintain_order, zlice) = options
        self.keep = Distinct._KEEP_MAP[keep]
        self.subset = set(subset) if subset is not None else None
        self.stable = maintain_order
        self.zlice = zlice

    def evaluate(self, *, cache: dict[int, DataFrame]) -> DataFrame:
        """Evaluate and return a dataframe."""
        df = self.df.evaluate(cache=cache)
        if self.subset is None:
            indices = list(range(df.num_columns))
        else:
            indices = [i for i, k in enumerate(df.names) if k in self.subset]
        keys_sorted = all(c.is_sorted for c in df.columns)
        if keys_sorted:
            table = plc.stream_compaction.unique(
                df.table,
                indices,
                self.keep,
                plc.types.NullEquality.EQUAL,
            )
        else:
            distinct = (
                plc.stream_compaction.stable_distinct
                if self.stable
                else plc.stream_compaction.distinct
            )
            table = distinct(
                df.table,
                indices,
                self.keep,
                plc.types.NullEquality.EQUAL,
                plc.types.NanEquality.ALL_EQUAL,
            )
        result = DataFrame(
            [Column(c, old.name) for c, old in zip(table.columns(), df.columns)], []
        )
        if keys_sorted or self.stable:
            result = result.with_sorted(like=df)
        return result.slice(self.zlice)


@dataclass(slots=True)
class Sort(IR):
    df: IR
    by: list[Expr]
    do_sort: Callable[..., plc.Table]
    zlice: tuple[int, int] | None
    order: list[plc.types.Order]
    null_order: list[plc.types.NullOrder]

    def __init__(self, schema: dict, df: IR, by: list[Expr], options: Any):
        self.schema = schema
        self.df = df
        self.by = by
        stable, nulls_last, descending = options
        self.order, self.null_order = sorting.sort_order(
            descending, nulls_last=nulls_last, num_keys=len(by)
        )
        self.do_sort = (
            plc.sorting.stable_sort_by_key if stable else plc.sorting.sort_by_key
        )

    def evaluate(self, *, cache: dict[int, DataFrame]) -> DataFrame:
        """Evaluate and return a dataframe."""
        df = self.df.evaluate(cache=cache)
        sort_keys = [k.evaluate(df) for k in self.by]
        keys_in_result = [
            i
            for k in sort_keys
            if (i := df.names.get(k.name)) is not None and k is df.columns[i]
        ]
        table = self.do_sort(
            df.table,
            plc.Table([k.obj for k in sort_keys]),
            self.order,
            self.null_order,
        )
        columns = [Column(c, old.name) for c, old in zip(table.columns(), df.columns)]
        # If a sort key is in the result table, set the sortedness property
        for idx in keys_in_result:
            columns[idx] = columns[idx].set_sorted(
                is_sorted=plc.types.Sorted.YES,
                order=self.order[idx],
                null_order=self.null_order[idx],
            )
        return DataFrame(columns, [])


@dataclass(slots=True)
class Slice(IR):
    df: IR
    offset: int
    length: int

    def evaluate(self, *, cache: dict[int, DataFrame]) -> DataFrame:
        """Evaluate and return a dataframe."""
        df = self.df.evaluate(cache=cache)
        return df.slice((self.offset, self.length))


@dataclass(slots=True)
class Filter(IR):
    df: IR
    mask: Expr

    def evaluate(self, *, cache: dict[int, DataFrame]) -> DataFrame:
        """Evaluate and return a dataframe."""
        df = self.df.evaluate(cache=cache)
        return df.filter(self.mask.evaluate(df))


@dataclass(slots=True)
class Projection(IR):
    df: IR

    def evaluate(self, *, cache: dict[int, DataFrame]) -> DataFrame:
        """Evaluate and return a dataframe."""
        df = self.df.evaluate(cache=cache)
        return df.select(set(self.schema.keys()))


@dataclass(slots=True)
class MapFunction(IR):
    df: IR
    name: str
    options: Any

    _NAMES: ClassVar[frozenset[str]] = frozenset(
        [
            "drop_nulls",
            "rechunk",
            "merge_sorted",
            "rename",
            "explode",
        ]
    )

    def __post_init__(self):
        """Validate preconditions."""
        if self.name not in MapFunction._NAMES:
            raise NotImplementedError(f"Unhandled map function {self.name}")
        if self.name == "explode":
            (to_explode,) = self.options
            if len(to_explode) > 1:
                # TODO: straightforward, but need to error check
                # polars requires that all to-explode columns have the
                # same sub-shapes
                raise NotImplementedError("Explode with more than one column")
        elif self.name == "merge_sorted":
            assert isinstance(self.df, Union)
            (key_column,) = self.options
            if key_column not in self.df.dfs[0].schema:
                raise ValueError(f"Key column {key_column} not found")

    def evaluate(self, *, cache: dict[int, DataFrame]) -> DataFrame:
        """Evaluate and return a dataframe."""
        if self.name == "merge_sorted":
            # merge_sorted operates on Union inputs
            # but if we evaluate the Union then we can't unpick the
            # pieces, so we dive inside and evaluate the pieces by hand
            assert isinstance(self.df, Union)
            first, *rest = (c.evaluate(cache=cache) for c in self.df.dfs)
            (key_column,) = self.options
            if not all(first.column_names == r.column_names for r in rest):
                raise ValueError("DataFrame shapes/column names don't match")
            # Already validated that key_column is in column names
            index = first.column_names.index(key_column)
            return DataFrame.from_table(
                plc.merge.merge_sorted(
                    [first.table, *(df.table for df in rest)],
                    [index],
                    [plc.types.Order.ASCENDING],
                    [plc.types.NullOrder.BEFORE],
                ),
                first.column_names,
            ).with_sorted(like=first, subset={key_column})
        elif self.name == "rechunk":
            # No-op in our data model
            return self.df.evaluate(cache=cache)
        elif self.name == "drop_nulls":
            df = self.df.evaluate(cache=cache)
            (subset,) = self.options
            subset = set(subset)
            indices = [i for i, name in enumerate(df.column_names) if name in subset]
            return DataFrame.from_table(
                plc.stream_compaction.drop_nulls(df.table, indices, len(indices)),
                df.column_names,
            ).with_sorted(like=df)
        elif self.name == "rename":
            df = self.df.evaluate(cache=cache)
            # final tag is "swapping" which is useful for the
            # optimiser (it blocks some pushdown operations)
            old, new, _ = self.options
            return df.rename_columns(dict(zip(old, new)))
        elif self.name == "explode":
            df = self.df.evaluate(cache=cache)
            ((to_explode,),) = self.options
            index = df.column_names.index(to_explode)
            subset = df.column_names_set - {to_explode}
            return DataFrame.from_table(
                plc.lists.explode_outer(df.table, index), df.column_names
            ).with_sorted(like=df, subset=subset)
        else:
            raise AssertionError("Should never be reached")


@dataclass(slots=True)
class Union(IR):
    dfs: list[IR]


@dataclass(slots=True)
class HConcat(IR):
    dfs: list[IR]


@dataclass(slots=True)
class ExtContext(IR):
    df: IR
    extra: list[IR]
