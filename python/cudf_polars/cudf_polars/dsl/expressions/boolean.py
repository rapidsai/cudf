# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
# TODO: remove need for this
# ruff: noqa: D101
"""Boolean DSL nodes."""

from __future__ import annotations

from enum import IntEnum, auto
from functools import partial, reduce
from typing import TYPE_CHECKING, Any, ClassVar

import pylibcudf as plc

from cudf_polars.containers import Column, DataType
from cudf_polars.dsl.expressions.base import (
    ExecutionContext,
    Expr,
)
from cudf_polars.utils.versions import POLARS_VERSION_LT_128

if TYPE_CHECKING:
    from typing_extensions import Self

    import polars.type_aliases as pl_types
    from polars.polars import _expr_nodes as pl_expr

    from cudf_polars.containers import DataFrame

__all__ = ["BooleanFunction"]


class BooleanFunction(Expr):
    class Name(IntEnum):
        """Internal and picklable representation of polars' `BooleanFunction`."""

        All = auto()
        AllHorizontal = auto()
        Any = auto()
        AnyHorizontal = auto()
        IsBetween = auto()
        IsDuplicated = auto()
        IsFinite = auto()
        IsFirstDistinct = auto()
        IsIn = auto()
        IsInfinite = auto()
        IsLastDistinct = auto()
        IsNan = auto()
        IsNotNan = auto()
        IsNotNull = auto()
        IsNull = auto()
        IsUnique = auto()
        Not = auto()

        @classmethod
        def from_polars(cls, obj: pl_expr.BooleanFunction) -> Self:
            """Convert from polars' `BooleanFunction`."""
            try:
                function, name = str(obj).split(".", maxsplit=1)
            except ValueError:
                # Failed to unpack string
                function = None
            if function != "BooleanFunction":
                raise ValueError("BooleanFunction required")
            return getattr(cls, name)

    __slots__ = ("name", "options")
    _non_child = ("dtype", "name", "options")

    def __init__(
        self,
        dtype: DataType,
        name: BooleanFunction.Name,
        options: tuple[Any, ...],
        *children: Expr,
    ) -> None:
        self.dtype = dtype
        self.options = options
        self.name = name
        self.children = children
        self.is_pointwise = self.name not in (
            BooleanFunction.Name.All,
            BooleanFunction.Name.Any,
            BooleanFunction.Name.IsDuplicated,
            BooleanFunction.Name.IsFirstDistinct,
            BooleanFunction.Name.IsLastDistinct,
            BooleanFunction.Name.IsUnique,
        )
        if (
            POLARS_VERSION_LT_128
            and self.name is BooleanFunction.Name.IsIn
            and not all(
                c.dtype.plc == self.children[0].dtype.plc for c in self.children
            )
        ):  # pragma: no cover
            # TODO: If polars IR doesn't put the casts in, we need to
            # mimic the supertype promotion rules.
            raise NotImplementedError("IsIn doesn't support supertype casting")

    @staticmethod
    def _distinct(
        column: Column,
        dtype: DataType,
        *,
        keep: plc.stream_compaction.DuplicateKeepOption,
        source_value: plc.Scalar,
        target_value: plc.Scalar,
    ) -> Column:
        table = plc.Table([column.obj])
        indices = plc.stream_compaction.distinct_indices(
            table,
            keep,
            # TODO: polars doesn't expose options for these
            plc.types.NullEquality.EQUAL,
            plc.types.NanEquality.ALL_EQUAL,
        )
        return Column(
            plc.copying.scatter(
                [source_value],
                indices,
                plc.Table([plc.Column.from_scalar(target_value, table.num_rows())]),
            ).columns()[0],
            dtype=dtype,
        )

    _BETWEEN_OPS: ClassVar[
        dict[
            pl_types.ClosedInterval,
            tuple[plc.binaryop.BinaryOperator, plc.binaryop.BinaryOperator],
        ]
    ] = {
        "none": (
            plc.binaryop.BinaryOperator.GREATER,
            plc.binaryop.BinaryOperator.LESS,
        ),
        "left": (
            plc.binaryop.BinaryOperator.GREATER_EQUAL,
            plc.binaryop.BinaryOperator.LESS,
        ),
        "right": (
            plc.binaryop.BinaryOperator.GREATER,
            plc.binaryop.BinaryOperator.LESS_EQUAL,
        ),
        "both": (
            plc.binaryop.BinaryOperator.GREATER_EQUAL,
            plc.binaryop.BinaryOperator.LESS_EQUAL,
        ),
    }

    def do_evaluate(
        self, df: DataFrame, *, context: ExecutionContext = ExecutionContext.FRAME
    ) -> Column:
        """Evaluate this expression given a dataframe for context."""
        if self.name in (
            BooleanFunction.Name.IsFinite,
            BooleanFunction.Name.IsInfinite,
        ):
            # Avoid evaluating the child if the dtype tells us it's unnecessary.
            (child,) = self.children
            is_finite = self.name is BooleanFunction.Name.IsFinite
            if child.dtype.id() not in (plc.TypeId.FLOAT32, plc.TypeId.FLOAT64):
                value = plc.Scalar.from_py(is_finite)
                return Column(
                    plc.Column.from_scalar(value, df.num_rows), dtype=self.dtype
                )
            needles = child.evaluate(df, context=context)
            to_search = [-float("inf"), float("inf")]
            if is_finite:
                # NaN is neither finite not infinite
                to_search.append(float("nan"))
            haystack = plc.Column.from_iterable_of_py(
                to_search,
                dtype=needles.obj.type(),
            )
            result = plc.search.contains(haystack, needles.obj)
            if is_finite:
                result = plc.unary.unary_operation(result, plc.unary.UnaryOperator.NOT)
            return Column(result, dtype=self.dtype)
        columns = [child.evaluate(df, context=context) for child in self.children]
        # Kleene logic for Any (OR) and All (AND) if ignore_nulls is
        # False
        if self.name in (BooleanFunction.Name.Any, BooleanFunction.Name.All):
            (ignore_nulls,) = self.options
            (column,) = columns
            is_any = self.name is BooleanFunction.Name.Any
            agg = plc.aggregation.any() if is_any else plc.aggregation.all()
            result = plc.reduce.reduce(column.obj, agg, self.dtype.plc)
            if not ignore_nulls and column.null_count > 0:
                #      Truth tables
                #     Any         All
                #   | F U T     | F U T
                # --+------   --+------
                # F | F U T   F | F F F
                # U | U U T   U | F U U
                # T | T T T   T | F U T
                #
                # If the input null count was non-zero, we must
                # post-process the result to insert the correct value.
                h_result = result.to_py()
                if (is_any and not h_result) or (not is_any and h_result):
                    # Any                     All
                    # False || Null => Null   True && Null => Null
                    return Column(
                        plc.Column.all_null_like(column.obj, 1), dtype=self.dtype
                    )
            return Column(plc.Column.from_scalar(result, 1), dtype=self.dtype)
        if self.name is BooleanFunction.Name.IsNull:
            (column,) = columns
            return Column(plc.unary.is_null(column.obj), dtype=self.dtype)
        elif self.name is BooleanFunction.Name.IsNotNull:
            (column,) = columns
            return Column(plc.unary.is_valid(column.obj), dtype=self.dtype)
        elif self.name is BooleanFunction.Name.IsNan:
            (column,) = columns
            return Column(
                plc.unary.is_nan(column.obj).with_mask(
                    column.obj.null_mask(), column.null_count
                ),
                dtype=self.dtype,
            )
        elif self.name is BooleanFunction.Name.IsNotNan:
            (column,) = columns
            return Column(
                plc.unary.is_not_nan(column.obj).with_mask(
                    column.obj.null_mask(), column.null_count
                ),
                dtype=self.dtype,
            )
        elif self.name is BooleanFunction.Name.IsFirstDistinct:
            (column,) = columns
            return self._distinct(
                column,
                dtype=self.dtype,
                keep=plc.stream_compaction.DuplicateKeepOption.KEEP_FIRST,
                source_value=plc.Scalar.from_py(py_val=True, dtype=self.dtype.plc),
                target_value=plc.Scalar.from_py(py_val=False, dtype=self.dtype.plc),
            )
        elif self.name is BooleanFunction.Name.IsLastDistinct:
            (column,) = columns
            return self._distinct(
                column,
                dtype=self.dtype,
                keep=plc.stream_compaction.DuplicateKeepOption.KEEP_LAST,
                source_value=plc.Scalar.from_py(py_val=True, dtype=self.dtype.plc),
                target_value=plc.Scalar.from_py(py_val=False, dtype=self.dtype.plc),
            )
        elif self.name is BooleanFunction.Name.IsUnique:
            (column,) = columns
            return self._distinct(
                column,
                dtype=self.dtype,
                keep=plc.stream_compaction.DuplicateKeepOption.KEEP_NONE,
                source_value=plc.Scalar.from_py(py_val=True, dtype=self.dtype.plc),
                target_value=plc.Scalar.from_py(py_val=False, dtype=self.dtype.plc),
            )
        elif self.name is BooleanFunction.Name.IsDuplicated:
            (column,) = columns
            return self._distinct(
                column,
                dtype=self.dtype,
                keep=plc.stream_compaction.DuplicateKeepOption.KEEP_NONE,
                source_value=plc.Scalar.from_py(py_val=False, dtype=self.dtype.plc),
                target_value=plc.Scalar.from_py(py_val=True, dtype=self.dtype.plc),
            )
        elif self.name is BooleanFunction.Name.AllHorizontal:
            return Column(
                reduce(
                    partial(
                        plc.binaryop.binary_operation,
                        op=plc.binaryop.BinaryOperator.NULL_LOGICAL_AND,
                        output_type=self.dtype.plc,
                    ),
                    (c.obj for c in columns),
                ),
                dtype=self.dtype,
            )
        elif self.name is BooleanFunction.Name.AnyHorizontal:
            return Column(
                reduce(
                    partial(
                        plc.binaryop.binary_operation,
                        op=plc.binaryop.BinaryOperator.NULL_LOGICAL_OR,
                        output_type=self.dtype.plc,
                    ),
                    (c.obj for c in columns),
                ),
                dtype=self.dtype,
            )
        elif self.name is BooleanFunction.Name.IsIn:
            needles, haystack = columns
            if haystack.obj.type().id() == plc.TypeId.LIST:
                # Unwrap values from the list column
                haystack = Column(
                    haystack.obj.children()[1],
                    dtype=DataType(haystack.dtype.polars.inner),
                ).astype(needles.dtype)
            if haystack.size:
                return Column(
                    plc.search.contains(haystack.obj, needles.obj), dtype=self.dtype
                )
            return Column(
                plc.Column.from_scalar(plc.Scalar.from_py(py_val=False), needles.size),
                dtype=self.dtype,
            )
        elif self.name is BooleanFunction.Name.Not:
            (column,) = columns
            return Column(
                plc.unary.unary_operation(column.obj, plc.unary.UnaryOperator.NOT),
                dtype=self.dtype,
            )
        else:
            raise NotImplementedError(
                f"BooleanFunction {self.name}"
            )  # pragma: no cover; handled by init raising
