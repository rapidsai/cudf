# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
# TODO: remove need for this
# ruff: noqa: D101
"""DSL nodes for aggregations."""

from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING, Any, ClassVar

import pyarrow as pa

import pylibcudf as plc

from cudf_polars.containers import Column
from cudf_polars.dsl.expressions.base import (
    AggInfo,
    ExecutionContext,
    Expr,
)
from cudf_polars.dsl.expressions.literal import Literal
from cudf_polars.dsl.expressions.unary import UnaryFunction

if TYPE_CHECKING:
    from collections.abc import Mapping

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
            req = plc.aggregation.count(null_handling=plc.types.NullPolicy.EXCLUDE)
        elif name == "quantile":
            _, quantile = self.children
            if not isinstance(quantile, Literal):
                raise NotImplementedError("Only support literal quantile values")
            req = plc.aggregation.quantile(
                quantiles=[quantile.value.as_py()], interp=Agg.interp_mapping[options]
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
        elif name in {"count", "first", "last"}:
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

    def collect_agg(self, *, depth: int) -> AggInfo:
        """Collect information about aggregations in groupbys."""
        if depth >= 1:
            raise NotImplementedError(
                "Nested aggregations in groupby"
            )  # pragma: no cover; check_agg trips first
        if (isminmax := self.name in {"min", "max"}) and self.options:
            raise NotImplementedError("Nan propagation in groupby for min/max")
        (child,) = self.children
        ((expr, _, _),) = child.collect_agg(depth=depth + 1).requests
        request = self.request
        # These are handled specially here because we don't set up the
        # request for the whole-frame agg because we can avoid a
        # reduce for these.
        if self.name == "first":
            request = plc.aggregation.nth_element(
                0, null_handling=plc.types.NullPolicy.INCLUDE
            )
        elif self.name == "last":
            request = plc.aggregation.nth_element(
                -1, null_handling=plc.types.NullPolicy.INCLUDE
            )
        if request is None:
            raise NotImplementedError(
                f"Aggregation {self.name} in groupby"
            )  # pragma: no cover; __init__ trips first
        if isminmax and plc.traits.is_floating_point(self.dtype):
            assert expr is not None
            # Ignore nans in these groupby aggs, do this by masking
            # nans in the input
            expr = UnaryFunction(self.dtype, "mask_nans", (), expr)
        return AggInfo([(expr, request, self)])

    def _reduce(
        self, column: Column, *, request: plc.aggregation.Aggregation
    ) -> Column:
        return Column(
            plc.Column.from_scalar(
                plc.reduce.reduce(column.obj, request, self.dtype),
                1,
            )
        )

    def _count(self, column: Column) -> Column:
        return Column(
            plc.Column.from_scalar(
                plc.interop.from_arrow(
                    pa.scalar(
                        column.obj.size() - column.obj.null_count(),
                        type=plc.interop.to_arrow(self.dtype),
                    ),
                ),
                1,
            )
        )

    def _min(self, column: Column, *, propagate_nans: bool) -> Column:
        if propagate_nans and column.nan_count > 0:
            return Column(
                plc.Column.from_scalar(
                    plc.interop.from_arrow(
                        pa.scalar(float("nan"), type=plc.interop.to_arrow(self.dtype))
                    ),
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
                    plc.interop.from_arrow(
                        pa.scalar(float("nan"), type=plc.interop.to_arrow(self.dtype))
                    ),
                    1,
                )
            )
        if column.nan_count > 0:
            column = column.mask_nans()
        return self._reduce(column, request=plc.aggregation.max())

    def _first(self, column: Column) -> Column:
        return Column(plc.copying.slice(column.obj, [0, 1])[0])

    def _last(self, column: Column) -> Column:
        n = column.obj.size()
        return Column(plc.copying.slice(column.obj, [n - 1, n])[0])

    def do_evaluate(
        self,
        df: DataFrame,
        *,
        context: ExecutionContext = ExecutionContext.FRAME,
        mapping: Mapping[Expr, Column] | None = None,
    ) -> Column:
        """Evaluate this expression given a dataframe for context."""
        if context is not ExecutionContext.FRAME:
            raise NotImplementedError(
                f"Agg in context {context}"
            )  # pragma: no cover; unreachable

        # Aggregations like quantiles may have additional children that were
        # preprocessed into pylibcudf requests.
        child = self.children[0]
        return self.op(child.evaluate(df, context=context, mapping=mapping))
