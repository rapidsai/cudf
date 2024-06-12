# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""
DSL nodes for the LogicalPlan of polars.

An IR node is either a source, normal, or a sink. Respectively they
can be considered as functions:

- source: `IO () -> DataFrame`
- normal: `DataFrame -> DataFrame`
- sink: `DataFrame -> IO ()`
"""

from __future__ import annotations

import dataclasses
import itertools
import types
from functools import cache
from typing import TYPE_CHECKING, Any, Callable, ClassVar, NoReturn

import pyarrow as pa
from typing_extensions import assert_never

import polars as pl

import cudf
import cudf._lib.pylibcudf as plc

import cudf_polars.dsl.expr as expr
from cudf_polars.containers import DataFrame, NamedColumn
from cudf_polars.utils import sorting

if TYPE_CHECKING:
    from collections.abc import MutableMapping
    from typing import Literal

    from cudf_polars.typing import Schema


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


def broadcast(
    *columns: NamedColumn, target_length: int | None = None
) -> list[NamedColumn]:
    """
    Broadcast a sequence of columns to a common length.

    Parameters
    ----------
    columns
        Columns to broadcast.
    target_length
        Optional length to broadcast to. If not provided, uses the
        non-unit length of existing columns.

    Returns
    -------
    List of broadcasted columns all of the same length.

    Raises
    ------
    RuntimeError
        If broadcasting is not possible.

    Notes
    -----
    In evaluation of a set of expressions, polars type-puns length-1
    columns with scalars. When we insert these into a DataFrame
    object, we need to ensure they are of equal length. This function
    takes some columns, some of which may be length-1 and ensures that
    all length-1 columns are broadcast to the length of the others.

    Broadcasting is only possible if the set of lengths of the input
    columns is a subset of ``{1, n}`` for some (fixed) ``n``. If
    ``target_length`` is provided and not all columns are length-1
    (i.e. ``n != 1``), then ``target_length`` must be equal to ``n``.
    """
    lengths: set[int] = {column.obj.size() for column in columns}
    if lengths == {1}:
        if target_length is None:
            return list(columns)
        nrows = target_length
    else:
        try:
            (nrows,) = lengths.difference([1])
        except ValueError as e:
            raise RuntimeError("Mismatching column lengths") from e
        if target_length is not None and nrows != target_length:
            raise RuntimeError(
                f"Cannot broadcast columns of length {nrows=} to {target_length=}"
            )
    return [
        column
        if column.obj.size() != 1
        else NamedColumn(
            plc.Column.from_scalar(column.obj_scalar, nrows),
            column.name,
            is_sorted=plc.types.Sorted.YES,
            order=plc.types.Order.ASCENDING,
            null_order=plc.types.NullOrder.BEFORE,
        )
        for column in columns
    ]


@dataclasses.dataclass(slots=True)
class IR:
    """Abstract plan node, representing an unevaluated dataframe."""

    schema: Schema
    """Mapping from column names to their data types."""

    def evaluate(self, *, cache: MutableMapping[int, DataFrame]) -> DataFrame:
        """
        Evaluate the node and return a dataframe.

        Parameters
        ----------
        cache
            Mapping from cached node ids to constructed DataFrames.
            Used to implement evaluation of the `Cache` node.

        Returns
        -------
        DataFrame (on device) representing the evaluation of this plan
        node.

        Raises
        ------
        NotImplementedError
            If we couldn't evaluate things. Ideally this should not occur,
            since the translation phase should pick up things that we
            cannot handle.
        """
        raise NotImplementedError


@dataclasses.dataclass(slots=True)
class PythonScan(IR):
    """Representation of input from a python function."""

    options: Any
    """Arbitrary options."""
    predicate: expr.NamedExpr | None
    """Filter to apply to the constructed dataframe before returning it."""


@dataclasses.dataclass(slots=True)
class Scan(IR):
    """Input from files."""

    typ: Any
    """What type of file are we reading? Parquet, CSV, etc..."""
    paths: list[str]
    """List of paths to read from."""
    file_options: Any
    """Options for reading the file.

    Attributes are:
    - ``with_columns: list[str]`` of projected columns to return.
    - ``n_rows: int``: Number of rows to read.
    - ``row_index: tuple[name, offset] | None``: Add an integer index
        column with given name.
    """
    predicate: expr.NamedExpr | None
    """Mask to apply to the read dataframe."""

    def __post_init__(self) -> None:
        """Validate preconditions."""
        if self.file_options.n_rows is not None:
            raise NotImplementedError("row limit in scan")
        if self.typ not in ("csv", "parquet"):
            raise NotImplementedError(
                f"Unhandled scan type: {self.typ}"
            )  # pragma: no cover; polars raises on the rust side for now

    def evaluate(self, *, cache: MutableMapping[int, DataFrame]) -> DataFrame:
        """Evaluate and return a dataframe."""
        options = self.file_options
        with_columns = options.with_columns
        row_index = options.row_index
        if self.typ == "csv":
            df = DataFrame.from_cudf(
                cudf.concat(
                    [cudf.read_csv(p, usecols=with_columns) for p in self.paths]
                )
            )
        elif self.typ == "parquet":
            cdf = cudf.read_parquet(self.paths, columns=with_columns)
            assert isinstance(cdf, cudf.DataFrame)
            df = DataFrame.from_cudf(cdf)
        else:
            assert_never(self.typ)
        if row_index is not None:
            name, offset = row_index
            dtype = self.schema[name]
            step = plc.interop.from_arrow(
                pa.scalar(1, type=plc.interop.to_arrow(dtype))
            )
            init = plc.interop.from_arrow(
                pa.scalar(offset, type=plc.interop.to_arrow(dtype))
            )
            index = NamedColumn(
                plc.filling.sequence(df.num_rows, init, step),
                name,
                is_sorted=plc.types.Sorted.YES,
                order=plc.types.Order.ASCENDING,
                null_order=plc.types.NullOrder.AFTER,
            )
            df = DataFrame([index, *df.columns])
        # TODO: should be true, but not the case until we get
        # cudf-classic out of the loop for IO since it converts date32
        # to datetime.
        # assert all(
        #     c.obj.type() == dtype
        #     for c, dtype in zip(df.columns, self.schema.values())
        # )
        if self.predicate is None:
            return df
        else:
            (mask,) = broadcast(self.predicate.evaluate(df), target_length=df.num_rows)
            return df.filter(mask)


@dataclasses.dataclass(slots=True)
class Cache(IR):
    """
    Return a cached plan node.

    Used for CSE at the plan level.
    """

    key: int
    """The cache key."""
    value: IR
    """The unevaluated node to cache."""

    def evaluate(self, *, cache: MutableMapping[int, DataFrame]) -> DataFrame:
        """Evaluate and return a dataframe."""
        try:
            return cache[self.key]
        except KeyError:
            return cache.setdefault(self.key, self.value.evaluate(cache=cache))


@dataclasses.dataclass(slots=True)
class DataFrameScan(IR):
    """
    Input from an existing polars DataFrame.

    This typically arises from ``q.collect().lazy()``
    """

    df: Any
    """Polars LazyFrame object."""
    projection: list[str]
    """List of columns to project out."""
    predicate: expr.NamedExpr | None
    """Mask to apply."""

    def evaluate(self, *, cache: MutableMapping[int, DataFrame]) -> DataFrame:
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
        assert all(
            c.obj.type() == dtype for c, dtype in zip(df.columns, self.schema.values())
        )
        if self.predicate is not None:
            (mask,) = broadcast(self.predicate.evaluate(df), target_length=df.num_rows)
            return df.filter(mask)
        else:
            return df


@dataclasses.dataclass(slots=True)
class Select(IR):
    """Produce a new dataframe selecting given expressions from an input."""

    df: IR
    """Input dataframe."""
    expr: list[expr.NamedExpr]
    """List of expressions to evaluate to form the new dataframe."""
    should_broadcast: bool
    """Should columns be broadcast?"""

    def evaluate(self, *, cache: MutableMapping[int, DataFrame]) -> DataFrame:
        """Evaluate and return a dataframe."""
        df = self.df.evaluate(cache=cache)
        # Handle any broadcasting
        columns = [e.evaluate(df) for e in self.expr]
        if self.should_broadcast:
            columns = broadcast(*columns)
        return DataFrame(columns)


@dataclasses.dataclass(slots=True)
class Reduce(IR):
    """
    Produce a new dataframe selecting given expressions from an input.

    This is a special case of :class:`Select` where all outputs are a single row.
    """

    df: IR
    """Input dataframe."""
    expr: list[expr.NamedExpr]
    """List of expressions to evaluate to form the new dataframe."""

    def evaluate(self, *, cache: MutableMapping[int, DataFrame]) -> DataFrame:
        """Evaluate and return a dataframe."""
        df = self.df.evaluate(cache=cache)
        columns = broadcast(*(e.evaluate(df) for e in self.expr))
        assert all(column.obj.size() == 1 for column in columns)
        return DataFrame(columns)


def placeholder_column(n: int) -> plc.Column:
    """
    Produce a placeholder pylibcudf column with NO BACKING DATA.

    Parameters
    ----------
    n
        Number of rows the column will advertise

    Returns
    -------
    pylibcudf Column that is almost unusable. DO NOT ACCESS THE DATA BUFFER.

    Notes
    -----
    This is used to avoid allocating data for count aggregations.
    """
    return plc.Column(
        plc.DataType(plc.TypeId.INT8),
        n,
        plc.gpumemoryview(
            types.SimpleNamespace(__cuda_array_interface__={"data": (1, True)})
        ),
        None,
        0,
        0,
        [],
    )


@dataclasses.dataclass(slots=False)
class GroupBy(IR):
    """Perform a groupby."""

    df: IR
    """Input dataframe."""
    agg_requests: list[expr.NamedExpr]
    """List of expressions to evaluate groupwise."""
    keys: list[expr.NamedExpr]
    """List of expressions forming the keys."""
    maintain_order: bool
    """Should the order of the input dataframe be maintained?"""
    options: Any
    """Options controlling style of groupby."""
    agg_infos: list[expr.AggInfo] = dataclasses.field(init=False)

    @staticmethod
    def check_agg(agg: expr.Expr) -> int:
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
        NotImplementedError
            For unsupported expression nodes.
        """
        if isinstance(agg, (expr.BinOp, expr.Cast)):
            return max(GroupBy.check_agg(child) for child in agg.children)
        elif isinstance(agg, expr.Agg):
            if agg.name == "implode":
                raise NotImplementedError("implode in groupby")
            return 1 + max(GroupBy.check_agg(child) for child in agg.children)
        elif isinstance(agg, (expr.Len, expr.Col, expr.Literal)):
            return 0
        else:
            raise NotImplementedError(f"No handler for {agg=}")

    def __post_init__(self) -> None:
        """Check whether all the aggregations are implemented."""
        if self.options.rolling is None and self.maintain_order:
            raise NotImplementedError("Maintaining order in groupby")
        if self.options.rolling:
            raise NotImplementedError("rolling window/groupby")
        if any(GroupBy.check_agg(a.value) > 1 for a in self.agg_requests):
            raise NotImplementedError("Nested aggregations in groupby")
        self.agg_infos = [req.collect_agg(depth=0) for req in self.agg_requests]

    def evaluate(self, *, cache: MutableMapping[int, DataFrame]) -> DataFrame:
        """Evaluate and return a dataframe."""
        df = self.df.evaluate(cache=cache)
        keys = broadcast(
            *(k.evaluate(df) for k in self.keys), target_length=df.num_rows
        )
        # TODO: use sorted information, need to expose column_order
        # and null_precedence in pylibcudf groupby constructor
        # sorted = (
        #     plc.types.Sorted.YES
        #     if all(k.is_sorted for k in keys)
        #     else plc.types.Sorted.NO
        # )
        grouper = plc.groupby.GroupBy(
            plc.Table([k.obj for k in keys]),
            null_handling=plc.types.NullPolicy.INCLUDE,
        )
        # TODO: uniquify
        requests = []
        replacements: list[expr.Expr] = []
        for info in self.agg_infos:
            for pre_eval, req, rep in info.requests:
                if pre_eval is None:
                    col = placeholder_column(df.num_rows)
                else:
                    col = pre_eval.evaluate(df).obj
                requests.append(plc.groupby.GroupByRequest(col, [req]))
                replacements.append(rep)
        group_keys, raw_tables = grouper.aggregate(requests)
        # TODO: names
        raw_columns: list[NamedColumn] = []
        for i, table in enumerate(raw_tables):
            (column,) = table.columns()
            raw_columns.append(NamedColumn(column, f"tmp{i}"))
        mapping = dict(zip(replacements, raw_columns))
        result_keys = [
            NamedColumn(gk, k.name) for gk, k in zip(group_keys.columns(), keys)
        ]
        result_subs = DataFrame(raw_columns)
        results = [
            req.evaluate(result_subs, mapping=mapping) for req in self.agg_requests
        ]
        return DataFrame([*result_keys, *results]).slice(self.options.slice)


@dataclasses.dataclass(slots=True)
class Join(IR):
    """A join of two dataframes."""

    left: IR
    """Left frame."""
    right: IR
    """Right frame."""
    left_on: list[expr.NamedExpr]
    """List of expressions used as keys in the left frame."""
    right_on: list[expr.NamedExpr]
    """List of expressions used as keys in the right frame."""
    options: tuple[
        Literal["inner", "left", "full", "leftsemi", "leftanti"],
        bool,
        tuple[int, int] | None,
        str | None,
        bool,
    ]
    """
    tuple of options:
    - how: join type
    - join_nulls: do nulls compare equal?
    - slice: optional slice to perform after joining.
    - suffix: string suffix for right columns if names match
    - coalesce: should key columns be coalesced (only makes sense for outer joins)
    """

    def __post_init__(self) -> None:
        """Validate preconditions."""
        if self.options[0] == "cross":
            raise NotImplementedError("cross join not implemented")

    @cache
    @staticmethod
    def _joiners(
        how: Literal["inner", "left", "full", "leftsemi", "leftanti"],
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
        elif how == "full":
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

    def evaluate(self, *, cache: MutableMapping[int, DataFrame]) -> DataFrame:
        """Evaluate and return a dataframe."""
        left = self.left.evaluate(cache=cache)
        right = self.right.evaluate(cache=cache)
        left_on = DataFrame(
            broadcast(
                *(e.evaluate(left) for e in self.left_on), target_length=left.num_rows
            )
        )
        right_on = DataFrame(
            broadcast(
                *(e.evaluate(right) for e in self.right_on),
                target_length=right.num_rows,
            )
        )
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
            lg, rg = join_fn(left_on.table, right_on.table, null_equality)
            left = left.replace_columns(*left_on.columns)
            right = right.replace_columns(*right_on.columns)
            if coalesce and how == "inner":
                right = right.discard_columns(right_on.column_names_set)
            left = DataFrame.from_table(
                plc.copying.gather(left.table, lg, left_policy), left.column_names
            )
            right = DataFrame.from_table(
                plc.copying.gather(right.table, rg, right_policy), right.column_names
            )
            if coalesce and how != "inner":
                left = left.replace_columns(
                    *(
                        NamedColumn(
                            plc.replace.replace_nulls(left_col.obj, right_col.obj),
                            left_col.name,
                        )
                        for left_col, right_col in zip(
                            left.select_columns(left_on.column_names_set),
                            right.select_columns(right_on.column_names_set),
                        )
                    )
                )
                right = right.discard_columns(right_on.column_names_set)
            right = right.rename_columns(
                {
                    name: f"{name}{suffix}"
                    for name in right.column_names
                    if name in left.column_names_set
                }
            )
            result = left.with_columns(right.columns)
        return result.slice(zlice)


@dataclasses.dataclass(slots=True)
class HStack(IR):
    """Add new columns to a dataframe."""

    df: IR
    """Input dataframe."""
    columns: list[expr.NamedExpr]
    """List of expressions to produce new columns."""
    should_broadcast: bool
    """Should columns be broadcast?"""

    def evaluate(self, *, cache: MutableMapping[int, DataFrame]) -> DataFrame:
        """Evaluate and return a dataframe."""
        df = self.df.evaluate(cache=cache)
        columns = [c.evaluate(df) for c in self.columns]
        if self.should_broadcast:
            columns = broadcast(*columns, target_length=df.num_rows)
        else:
            # Polars ensures this is true, but let's make sure nothing
            # went wrong. In this case, the parent node is a
            # guaranteed to be a Select which will take care of making
            # sure that everything is the same length. The result
            # table that might have mismatching column lengths will
            # never be turned into a pylibcudf Table with all columns
            # by the Select, which is why this is safe.
            assert all(e.name.startswith("__POLARS_CSER_0x") for e in self.columns)
        return df.with_columns(columns)


@dataclasses.dataclass(slots=True)
class Distinct(IR):
    """Produce a new dataframe with distinct rows."""

    df: IR
    """Input dataframe."""
    keep: plc.stream_compaction.DuplicateKeepOption
    """Which rows to keep."""
    subset: set[str] | None
    """Which columns to inspect when computing distinct rows."""
    zlice: tuple[int, int] | None
    """Optional slice to perform after compaction."""
    stable: bool
    """Should order be preserved?"""

    _KEEP_MAP: ClassVar[dict[str, plc.stream_compaction.DuplicateKeepOption]] = {
        "first": plc.stream_compaction.DuplicateKeepOption.KEEP_FIRST,
        "last": plc.stream_compaction.DuplicateKeepOption.KEEP_LAST,
        "none": plc.stream_compaction.DuplicateKeepOption.KEEP_NONE,
        "any": plc.stream_compaction.DuplicateKeepOption.KEEP_ANY,
    }

    def __init__(self, schema: Schema, df: IR, options: Any) -> None:
        self.schema = schema
        self.df = df
        (keep, subset, maintain_order, zlice) = options
        self.keep = Distinct._KEEP_MAP[keep]
        self.subset = set(subset) if subset is not None else None
        self.stable = maintain_order
        self.zlice = zlice

    def evaluate(self, *, cache: MutableMapping[int, DataFrame]) -> DataFrame:
        """Evaluate and return a dataframe."""
        df = self.df.evaluate(cache=cache)
        if self.subset is None:
            indices = list(range(df.num_columns))
        else:
            indices = [i for i, k in enumerate(df.column_names) if k in self.subset]
        keys_sorted = all(df.columns[i].is_sorted for i in indices)
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
            [
                NamedColumn(c, old.name).sorted_like(old)
                for c, old in zip(table.columns(), df.columns)
            ]
        )
        if keys_sorted or self.stable:
            result = result.sorted_like(df)
        return result.slice(self.zlice)


@dataclasses.dataclass(slots=True)
class Sort(IR):
    """Sort a dataframe."""

    df: IR
    """Input."""
    by: list[expr.NamedExpr]
    """List of expressions to produce sort keys."""
    do_sort: Callable[..., plc.Table]
    """pylibcudf sorting function."""
    zlice: tuple[int, int] | None
    """Optional slice to apply after sorting."""
    order: list[plc.types.Order]
    """Order keys should be sorted in."""
    null_order: list[plc.types.NullOrder]
    """Where nulls sort to."""

    def __init__(
        self,
        schema: Schema,
        df: IR,
        by: list[expr.NamedExpr],
        options: Any,
        zlice: tuple[int, int] | None,
    ) -> None:
        self.schema = schema
        self.df = df
        self.by = by
        self.zlice = zlice
        stable, nulls_last, descending = options
        self.order, self.null_order = sorting.sort_order(
            descending, nulls_last=nulls_last, num_keys=len(by)
        )
        self.do_sort = (
            plc.sorting.stable_sort_by_key if stable else plc.sorting.sort_by_key
        )

    def evaluate(self, *, cache: MutableMapping[int, DataFrame]) -> DataFrame:
        """Evaluate and return a dataframe."""
        df = self.df.evaluate(cache=cache)
        sort_keys = broadcast(
            *(k.evaluate(df) for k in self.by), target_length=df.num_rows
        )
        names = {c.name: i for i, c in enumerate(df.columns)}
        # TODO: More robust identification here.
        keys_in_result = [
            i
            for k in sort_keys
            if (i := names.get(k.name)) is not None and k.obj is df.columns[i].obj
        ]
        table = self.do_sort(
            df.table,
            plc.Table([k.obj for k in sort_keys]),
            self.order,
            self.null_order,
        )
        columns = [
            NamedColumn(c, old.name) for c, old in zip(table.columns(), df.columns)
        ]
        # If a sort key is in the result table, set the sortedness property
        for k, i in enumerate(keys_in_result):
            columns[i] = columns[i].set_sorted(
                is_sorted=plc.types.Sorted.YES,
                order=self.order[k],
                null_order=self.null_order[k],
            )
        return DataFrame(columns).slice(self.zlice)


@dataclasses.dataclass(slots=True)
class Slice(IR):
    """Slice a dataframe."""

    df: IR
    """Input."""
    offset: int
    """Start of the slice."""
    length: int
    """Length of the slice."""

    def evaluate(self, *, cache: MutableMapping[int, DataFrame]) -> DataFrame:
        """Evaluate and return a dataframe."""
        df = self.df.evaluate(cache=cache)
        return df.slice((self.offset, self.length))


@dataclasses.dataclass(slots=True)
class Filter(IR):
    """Filter a dataframe with a boolean mask."""

    df: IR
    """Input."""
    mask: expr.NamedExpr
    """Expression evaluating to a mask."""

    def evaluate(self, *, cache: MutableMapping[int, DataFrame]) -> DataFrame:
        """Evaluate and return a dataframe."""
        df = self.df.evaluate(cache=cache)
        (mask,) = broadcast(self.mask.evaluate(df), target_length=df.num_rows)
        return df.filter(mask)


@dataclasses.dataclass(slots=True)
class Projection(IR):
    """Select a subset of columns from a dataframe."""

    df: IR
    """Input."""

    def evaluate(self, *, cache: MutableMapping[int, DataFrame]) -> DataFrame:
        """Evaluate and return a dataframe."""
        df = self.df.evaluate(cache=cache)
        # This can reorder things.
        columns = broadcast(
            *df.select(list(self.schema.keys())).columns, target_length=df.num_rows
        )
        return DataFrame(columns)


@dataclasses.dataclass(slots=True)
class MapFunction(IR):
    """Apply some function to a dataframe."""

    df: IR
    """Input."""
    name: str
    """Function name."""
    options: Any
    """Arbitrary options, interpreted per function."""

    _NAMES: ClassVar[frozenset[str]] = frozenset(
        [
            "drop_nulls",
            "rechunk",
            "merge_sorted",
            "rename",
            "explode",
        ]
    )

    def __post_init__(self) -> None:
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

    def evaluate(self, *, cache: MutableMapping[int, DataFrame]) -> DataFrame:
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
            ).sorted_like(first, subset={key_column})
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
            ).sorted_like(df)
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
            ).sorted_like(df, subset=subset)
        else:
            raise AssertionError("Should never be reached")


@dataclasses.dataclass(slots=True)
class Union(IR):
    """Concatenate dataframes vertically."""

    dfs: list[IR]
    """List of inputs."""
    zlice: tuple[int, int] | None
    """Optional slice to apply after concatenation."""

    def __post_init__(self) -> None:
        """Validated preconditions."""
        schema = self.dfs[0].schema
        if not all(s.schema == schema for s in self.dfs[1:]):
            raise ValueError("Schema mismatch")

    def evaluate(self, *, cache: MutableMapping[int, DataFrame]) -> DataFrame:
        """Evaluate and return a dataframe."""
        # TODO: only evaluate what we need if we have a slice
        dfs = [df.evaluate(cache=cache) for df in self.dfs]
        return DataFrame.from_table(
            plc.concatenate.concatenate([df.table for df in dfs]), dfs[0].column_names
        ).slice(self.zlice)


@dataclasses.dataclass(slots=True)
class HConcat(IR):
    """Concatenate dataframes horizontally."""

    dfs: list[IR]
    """List of inputs."""

    def evaluate(self, *, cache: MutableMapping[int, DataFrame]) -> DataFrame:
        """Evaluate and return a dataframe."""
        dfs = [df.evaluate(cache=cache) for df in self.dfs]
        return DataFrame(
            list(itertools.chain.from_iterable(df.columns for df in dfs)),
        )


@dataclasses.dataclass(slots=True)
class ExtContext(IR):
    """
    Concatenate dataframes horizontally.

    Prefer HConcat, since this is going to be deprecated on the polars side.
    """

    df: IR
    """Input."""
    extra: list[IR]
    """List of extra inputs."""

    def __post_init__(self) -> NoReturn:
        """Validate preconditions."""
        raise NotImplementedError(
            "ExtContext will be deprecated, use horizontal concat instead."
        )
