# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
# TODO: remove need for this
# ruff: noqa: D101
"""DSL nodes for aggregations."""

from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING, Any, ClassVar

import pylibcudf as plc

from cudf_polars.containers import Column
from cudf_polars.dsl.expressions.base import ExecutionContext, Expr
from cudf_polars.dsl.expressions.literal import Literal

if TYPE_CHECKING:
    from cudf_polars.containers import DataFrame

__all__ = ["Agg"]


class Agg(Expr):
    __slots__ = ("name", "op", "options", "request")
    _non_child = ("dtype", "name", "options")

    def __init__(
        self, dtype: plc.DataType, name: str, options: Any, *children: Expr
    ) -> None:
        self.dtype = dtype
        self.name = name
        self.options = options
        self.is_pointwise = False
        self.children = children
        if name not in Agg._SUPPORTED:
            raise NotImplementedError(
                f"Unsupported aggregation {name=}"
            )  # pragma: no cover; all valid aggs are supported
        # TODO: nan handling in groupby case
        if name == "min":
            req = plc.aggregation.min()
        elif name == "max":
            req = plc.aggregation.max()
        elif name == "median":
            req = plc.aggregation.median()
        elif name == "n_unique":
            # TODO: datatype of result
            req = plc.aggregation.nunique(null_handling=plc.types.NullPolicy.INCLUDE)
        elif name == "first" or name == "last":
            req = None
        elif name == "mean":
            req = plc.aggregation.mean()
        elif name == "sum":
            req = plc.aggregation.sum()
        elif name == "std":
            # TODO: handle nans
            req = plc.aggregation.std(ddof=options)
        elif name == "var":
            # TODO: handle nans
            req = plc.aggregation.variance(ddof=options)
        elif name == "count":
            req = plc.aggregation.count(
                null_handling=plc.types.NullPolicy.EXCLUDE
                if not options
                else plc.types.NullPolicy.INCLUDE
            )
        elif name == "quantile":
            child, quantile = self.children
            if not isinstance(quantile, Literal):
                raise NotImplementedError("Only support literal quantile values")
            if options == "equiprobable":
                raise NotImplementedError("Quantile with equiprobable interpolation")
            if plc.traits.is_duration(child.dtype):
                raise NotImplementedError("Quantile with duration data type")
            req = plc.aggregation.quantile(
                quantiles=[quantile.value], interp=Agg.interp_mapping[options]
            )
        else:
            raise NotImplementedError(
                f"Unreachable, {name=} is incorrectly listed in _SUPPORTED"
            )  # pragma: no cover
        self.request = req
        op = getattr(self, f"_{name}", None)
        if op is None:
            op = partial(self._reduce, request=req)
        elif name in {"min", "max"}:
            op = partial(op, propagate_nans=options)
        elif name == "count":
            op = partial(op, include_nulls=options)
        elif name in {"sum", "first", "last"}:
            pass
        else:
            raise NotImplementedError(
                f"Unreachable, supported agg {name=} has no implementation"
            )  # pragma: no cover
        self.op = op

    _SUPPORTED: ClassVar[frozenset[str]] = frozenset(
        [
            "min",
            "max",
            "median",
            "n_unique",
            "first",
            "last",
            "mean",
            "sum",
            "count",
            "std",
            "var",
            "quantile",
        ]
    )

    interp_mapping: ClassVar[dict[str, plc.types.Interpolation]] = {
        "nearest": plc.types.Interpolation.NEAREST,
        "higher": plc.types.Interpolation.HIGHER,
        "lower": plc.types.Interpolation.LOWER,
        "midpoint": plc.types.Interpolation.MIDPOINT,
        "linear": plc.types.Interpolation.LINEAR,
    }

    @property
    def agg_request(self) -> plc.aggregation.Aggregation:  # noqa: D102
        if self.name == "first":
            return plc.aggregation.nth_element(
                0, null_handling=plc.types.NullPolicy.INCLUDE
            )
        elif self.name == "last":
            return plc.aggregation.nth_element(
                -1, null_handling=plc.types.NullPolicy.INCLUDE
            )
        else:
            assert self.request is not None, "Init should have raised"
            return self.request

    def _reduce(
        self, column: Column, *, request: plc.aggregation.Aggregation
    ) -> Column:
        return Column(
            plc.Column.from_scalar(
                plc.reduce.reduce(column.obj, request, self.dtype),
                1,
            )
        )

    def _count(self, column: Column, *, include_nulls: bool) -> Column:
        null_count = column.null_count if not include_nulls else 0
        return Column(
            plc.Column.from_scalar(
                plc.Scalar.from_py(column.size - null_count, self.dtype),
                1,
            )
        )

    def _sum(self, column: Column) -> Column:
        if column.size == 0 or column.null_count == column.size:
            return Column(
                plc.Column.from_scalar(
                    plc.Scalar.from_py(0, self.dtype),
                    1,
                )
            )
        return self._reduce(column, request=plc.aggregation.sum())

    def _min(self, column: Column, *, propagate_nans: bool) -> Column:
        if propagate_nans and column.nan_count > 0:
            return Column(
                plc.Column.from_scalar(
                    plc.Scalar.from_py(float("nan"), self.dtype),
                    1,
                )
            )
        if column.nan_count > 0:
            column = column.mask_nans()
        return self._reduce(column, request=plc.aggregation.min())

    def _max(self, column: Column, *, propagate_nans: bool) -> Column:
        if propagate_nans and column.nan_count > 0:
            return Column(
                plc.Column.from_scalar(
                    plc.Scalar.from_py(float("nan"), self.dtype),
                    1,
                )
            )
        if column.nan_count > 0:
            column = column.mask_nans()
        return self._reduce(column, request=plc.aggregation.max())

    def _first(self, column: Column) -> Column:
        return Column(plc.copying.slice(column.obj, [0, 1])[0])

    def _last(self, column: Column) -> Column:
        n = column.size
        return Column(plc.copying.slice(column.obj, [n - 1, n])[0])

    def do_evaluate(
        self, df: DataFrame, *, context: ExecutionContext = ExecutionContext.FRAME
    ) -> Column:
        """Evaluate this expression given a dataframe for context."""
        if context is not ExecutionContext.FRAME:
            raise NotImplementedError(
                f"Agg in context {context}"
            )  # pragma: no cover; unreachable

        # Aggregations like quantiles may have additional children that were
        # preprocessed into pylibcudf requests.
        child = self.children[0]
        return self.op(child.evaluate(df, context=context))
