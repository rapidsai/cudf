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
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, ClassVar

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
    if len(columns) == 0:
        return []
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


@dataclasses.dataclass
class IR:
    """Abstract plan node, representing an unevaluated dataframe."""

    schema: Schema
    """Mapping from column names to their data types."""

    def __post_init__(self):
        """Validate preconditions."""
        if any(dtype.id() == plc.TypeId.EMPTY for dtype in self.schema.values()):
            raise NotImplementedError("Cannot make empty columns.")

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
        raise NotImplementedError(
            f"Evaluation of plan {type(self).__name__}"
        )  # pragma: no cover


@dataclasses.dataclass
class PythonScan(IR):
    """Representation of input from a python function."""

    options: Any
    """Arbitrary options."""
    predicate: expr.NamedExpr | None
    """Filter to apply to the constructed dataframe before returning it."""

    def __post_init__(self):
        """Validate preconditions."""
        raise NotImplementedError("PythonScan not implemented")


@dataclasses.dataclass
class Scan(IR):
    """Input from files."""

    typ: str
    """What type of file are we reading? Parquet, CSV, etc..."""
    reader_options: dict[str, Any]
    """Reader-specific options, as dictionary."""
    cloud_options: dict[str, Any] | None
    """Cloud-related authentication options, currently ignored."""
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
        if self.typ not in ("csv", "parquet", "ndjson"):  # pragma: no cover
            # This line is unhittable ATM since IPC/Anonymous scan raise
            # on the polars side
            raise NotImplementedError(f"Unhandled scan type: {self.typ}")
        if self.cloud_options is not None and any(
            self.cloud_options[k] is not None for k in ("aws", "azure", "gcp")
        ):
            raise NotImplementedError(
                "Read from cloud storage"
            )  # pragma: no cover; no test yet
        if self.typ == "csv":
            if self.reader_options["skip_rows_after_header"] != 0:
                raise NotImplementedError("Skipping rows after header in CSV reader")
            parse_options = self.reader_options["parse_options"]
            if (
                null_values := parse_options["null_values"]
            ) is not None and "Named" in null_values:
                raise NotImplementedError(
                    "Per column null value specification not supported for CSV reader"
                )
            if (
                comment := parse_options["comment_prefix"]
            ) is not None and "Multi" in comment:
                raise NotImplementedError(
                    "Multi-character comment prefix not supported for CSV reader"
                )
            if not self.reader_options["has_header"]:
                # Need to do some file introspection to get the number
                # of columns so that column projection works right.
                raise NotImplementedError("Reading CSV without header")
        elif self.typ == "ndjson":
            # TODO: consider handling the low memory option here
            # (maybe use chunked JSON reader)
            if self.reader_options["infer_schema_length"] != 100:
                raise NotImplementedError(
                    "infer_schema_length is not supported in the JSON reader"
                )
            if self.reader_options["ignore_errors"]:
                raise NotImplementedError(
                    "ignore_errors is not supported in the JSON reader"
                )

    def evaluate(self, *, cache: MutableMapping[int, DataFrame]) -> DataFrame:
        """Evaluate and return a dataframe."""
        options = self.file_options
        with_columns = options.with_columns
        row_index = options.row_index
        if self.typ == "csv":
            dtype_map = {
                name: cudf._lib.types.PYLIBCUDF_TO_SUPPORTED_NUMPY_TYPES[typ.id()]
                for name, typ in self.schema.items()
            }
            parse_options = self.reader_options["parse_options"]
            sep = chr(parse_options["separator"])
            quote = chr(parse_options["quote_char"])
            eol = chr(parse_options["eol_char"])
            if self.reader_options["schema"] is not None:
                # Reader schema provides names
                column_names = list(self.reader_options["schema"]["inner"].keys())
            else:
                # file provides column names
                column_names = None
            usecols = with_columns
            # TODO: support has_header=False
            header = 0

            # polars defaults to no null recognition
            null_values = [""]
            if parse_options["null_values"] is not None:
                ((typ, nulls),) = parse_options["null_values"].items()
                if typ == "AllColumnsSingle":
                    # Single value
                    null_values.append(nulls)
                else:
                    # List of values
                    null_values.extend(nulls)
            if parse_options["comment_prefix"] is not None:
                comment = chr(parse_options["comment_prefix"]["Single"])
            else:
                comment = None
            decimal = "," if parse_options["decimal_comma"] else "."

            # polars skips blank lines at the beginning of the file
            pieces = []
            for p in self.paths:
                skiprows = self.reader_options["skip_rows"]
                # TODO: read_csv expands globs which we should not do,
                # because polars will already have handled them.
                path = Path(p)
                with path.open() as f:
                    while f.readline() == "\n":
                        skiprows += 1
                pieces.append(
                    cudf.read_csv(
                        path,
                        sep=sep,
                        quotechar=quote,
                        lineterminator=eol,
                        names=column_names,
                        header=header,
                        usecols=usecols,
                        na_filter=True,
                        na_values=null_values,
                        keep_default_na=False,
                        skiprows=skiprows,
                        comment=comment,
                        decimal=decimal,
                        dtype=dtype_map,
                    )
                )
            df = DataFrame.from_cudf(cudf.concat(pieces))
        elif self.typ == "parquet":
            cdf = cudf.read_parquet(self.paths, columns=with_columns)
            assert isinstance(cdf, cudf.DataFrame)
            df = DataFrame.from_cudf(cdf)
        elif self.typ == "ndjson":
            json_schema: list[tuple[str, str, list]] = [
                (name, typ, []) for name, typ in self.schema.items()
            ]
            plc_tbl_w_meta = plc.io.json.read_json(
                plc.io.SourceInfo(self.paths),
                lines=True,
                dtypes=json_schema,
                prune_columns=True,
            )
            # TODO: I don't think cudf-polars supports nested types in general right now
            # (but when it does, we should pass child column names from nested columns in)
            df = DataFrame.from_table(
                plc_tbl_w_meta.tbl, plc_tbl_w_meta.column_names(include_children=False)
            )
            # TODO: libcudf doesn't support column-projection (like usecols)
            # We should change the prune_columns param to usecols there to support this
            if with_columns is not None:
                df = df.select(with_columns)
        else:
            raise NotImplementedError(
                f"Unhandled scan type: {self.typ}"
            )  # pragma: no cover; post init trips first
        if (
            row_index is not None
            # TODO: remove condition when dropping support for polars 1.0
            # https://github.com/pola-rs/polars/pull/17363
            and row_index[0] in self.schema
        ):
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


@dataclasses.dataclass
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


@dataclasses.dataclass
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
        df = DataFrame.from_polars(pdf)
        assert all(
            c.obj.type() == dtype for c, dtype in zip(df.columns, self.schema.values())
        )
        if self.predicate is not None:
            (mask,) = broadcast(self.predicate.evaluate(df), target_length=df.num_rows)
            return df.filter(mask)
        else:
            return df


@dataclasses.dataclass
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


@dataclasses.dataclass
class Reduce(IR):
    """
    Produce a new dataframe selecting given expressions from an input.

    This is a special case of :class:`Select` where all outputs are a single row.
    """

    df: IR
    """Input dataframe."""
    expr: list[expr.NamedExpr]
    """List of expressions to evaluate to form the new dataframe."""

    def evaluate(
        self, *, cache: MutableMapping[int, DataFrame]
    ) -> DataFrame:  # pragma: no cover; polars doesn't emit this node yet
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


@dataclasses.dataclass
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
        if isinstance(agg, (expr.BinOp, expr.Cast, expr.UnaryFunction)):
            return max(GroupBy.check_agg(child) for child in agg.children)
        elif isinstance(agg, expr.Agg):
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
            raise NotImplementedError(
                "rolling window/groupby"
            )  # pragma: no cover; rollingwindow constructor has already raised
        if any(GroupBy.check_agg(a.value) > 1 for a in self.agg_requests):
            raise NotImplementedError("Nested aggregations in groupby")
        self.agg_infos = [req.collect_agg(depth=0) for req in self.agg_requests]

    def evaluate(self, *, cache: MutableMapping[int, DataFrame]) -> DataFrame:
        """Evaluate and return a dataframe."""
        df = self.df.evaluate(cache=cache)
        keys = broadcast(
            *(k.evaluate(df) for k in self.keys), target_length=df.num_rows
        )
        sorted = (
            plc.types.Sorted.YES
            if all(k.is_sorted for k in keys)
            else plc.types.Sorted.NO
        )
        grouper = plc.groupby.GroupBy(
            plc.Table([k.obj for k in keys]),
            null_handling=plc.types.NullPolicy.INCLUDE,
            keys_are_sorted=sorted,
            column_order=[k.order for k in keys],
            null_precedence=[k.null_order for k in keys],
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


@dataclasses.dataclass
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
        Literal["inner", "left", "full", "leftsemi", "leftanti", "cross"],
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
        if any(
            isinstance(e.value, expr.Literal)
            for e in itertools.chain(self.left_on, self.right_on)
        ):
            raise NotImplementedError("Join with literal as join key.")

    @staticmethod
    @cache
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
        how, join_nulls, zlice, suffix, coalesce = self.options
        suffix = "_right" if suffix is None else suffix
        if how == "cross":
            # Separate implementation, since cross_join returns the
            # result, not the gather maps
            columns = plc.join.cross_join(left.table, right.table).columns()
            left_cols = [
                NamedColumn(new, old.name).sorted_like(old)
                for new, old in zip(columns[: left.num_columns], left.columns)
            ]
            right_cols = [
                NamedColumn(
                    new,
                    old.name
                    if old.name not in left.column_names_set
                    else f"{old.name}{suffix}",
                )
                for new, old in zip(columns[left.num_columns :], right.columns)
            ]
            return DataFrame([*left_cols, *right_cols])
        # TODO: Waiting on clarity based on https://github.com/pola-rs/polars/issues/17184
        left_on = DataFrame(broadcast(*(e.evaluate(left) for e in self.left_on)))
        right_on = DataFrame(broadcast(*(e.evaluate(right) for e in self.right_on)))
        null_equality = (
            plc.types.NullEquality.EQUAL
            if join_nulls
            else plc.types.NullEquality.UNEQUAL
        )
        join_fn, left_policy, right_policy = Join._joiners(how)
        if right_policy is None:
            # Semi join
            lg = join_fn(left_on.table, right_on.table, null_equality)
            table = plc.copying.gather(left.table, lg, left_policy)
            result = DataFrame.from_table(table, left.column_names)
        else:
            lg, rg = join_fn(left_on.table, right_on.table, null_equality)
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


@dataclasses.dataclass
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


@dataclasses.dataclass
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


@dataclasses.dataclass
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


@dataclasses.dataclass
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


@dataclasses.dataclass
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


@dataclasses.dataclass
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


@dataclasses.dataclass
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
            "rechunk",
            # libcudf merge is not stable wrt order of inputs, since
            # it uses a priority queue to manage the tables it produces.
            # See: https://github.com/rapidsai/cudf/issues/16010
            # "merge_sorted",
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
        elif self.name == "rename":
            old, new, _ = self.options
            # TODO: perhaps polars should validate renaming in the IR?
            if len(new) != len(set(new)) or (
                set(new) & (set(self.df.schema.keys() - set(old)))
            ):
                raise NotImplementedError("Duplicate new names in rename.")

    def evaluate(self, *, cache: MutableMapping[int, DataFrame]) -> DataFrame:
        """Evaluate and return a dataframe."""
        if self.name == "rechunk":
            # No-op in our data model
            # Don't think this appears in a plan tree from python
            return self.df.evaluate(cache=cache)  # pragma: no cover
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
            raise AssertionError("Should never be reached")  # pragma: no cover


@dataclasses.dataclass
class Union(IR):
    """Concatenate dataframes vertically."""

    dfs: list[IR]
    """List of inputs."""
    zlice: tuple[int, int] | None
    """Optional slice to apply after concatenation."""

    def __post_init__(self) -> None:
        """Validate preconditions."""
        schema = self.dfs[0].schema
        if not all(s.schema == schema for s in self.dfs[1:]):
            raise NotImplementedError("Schema mismatch")

    def evaluate(self, *, cache: MutableMapping[int, DataFrame]) -> DataFrame:
        """Evaluate and return a dataframe."""
        # TODO: only evaluate what we need if we have a slice
        dfs = [df.evaluate(cache=cache) for df in self.dfs]
        return DataFrame.from_table(
            plc.concatenate.concatenate([df.table for df in dfs]), dfs[0].column_names
        ).slice(self.zlice)


@dataclasses.dataclass
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
