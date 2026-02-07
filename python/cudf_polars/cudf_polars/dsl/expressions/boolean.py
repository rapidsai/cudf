# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
# TODO: Document BooleanFunction to remove noqa
# ruff: noqa: D101
"""Boolean DSL nodes."""

from __future__ import annotations

from enum import IntEnum, auto
from functools import partial, reduce
from typing import TYPE_CHECKING, Any, ClassVar, cast

import polars as pl

import pylibcudf as plc

from cudf_polars.containers import Column, DataType
from cudf_polars.dsl.expressions.base import (
    ExecutionContext,
    Expr,
)

if TYPE_CHECKING:
    from typing import Self

    import polars.type_aliases as pl_types
    from polars import polars  # type: ignore[attr-defined]

    from rmm.pylibrmm.stream import Stream

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
        IsClose = auto()
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
        def from_polars(cls, obj: polars._expr_nodes.BooleanFunction) -> Self:
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
        if self.name in {
            BooleanFunction.Name.IsClose,
        }:
            raise NotImplementedError(
                f"Boolean function {self.name}"
            )  # pragma: no cover

    @staticmethod
    def _distinct(
        column: Column,
        dtype: DataType,
        *,
        keep: plc.stream_compaction.DuplicateKeepOption,
        source_value: plc.Scalar,
        target_value: plc.Scalar,
        stream: Stream,
    ) -> Column:
        table = plc.Table([column.obj])
        indices = plc.stream_compaction.distinct_indices(
            table,
            keep,
            # TODO: polars doesn't expose options for these
            plc.types.NullEquality.EQUAL,
            plc.types.NanEquality.ALL_EQUAL,
            stream=stream,
        )
        return Column(
            plc.copying.scatter(
                [source_value],
                indices,
                plc.Table(
                    [
                        plc.Column.from_scalar(
                            target_value, table.num_rows(), stream=stream
                        )
                    ]
                ),
                stream=stream,
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
            values = child.evaluate(df, context=context)
            is_float = values.obj.type().id() in (
                plc.TypeId.FLOAT32,
                plc.TypeId.FLOAT64,
            )
            is_finite = self.name is BooleanFunction.Name.IsFinite
            if not is_float:
                base = plc.Column.from_scalar(
                    plc.Scalar.from_py(py_val=is_finite, stream=df.stream),
                    values.size,
                    stream=df.stream,
                )
                out = base.with_mask(values.obj.null_mask(), values.null_count)
                return Column(out, dtype=self.dtype)
            to_search = [-float("inf"), float("inf")]
            if is_finite:
                # NaN is neither finite not infinite
                to_search.append(float("nan"))
            nonfinite_values = plc.Column.from_iterable_of_py(
                to_search,
                dtype=values.obj.type(),
                stream=df.stream,
            )
            result = plc.search.contains(nonfinite_values, values.obj, stream=df.stream)
            if is_finite:
                result = plc.unary.unary_operation(
                    result, plc.unary.UnaryOperator.NOT, stream=df.stream
                )
            return Column(
                result.with_mask(values.obj.null_mask(), values.null_count),
                dtype=self.dtype,
            )
        columns = [child.evaluate(df, context=context) for child in self.children]
        # Kleene logic for Any (OR) and All (AND) if ignore_nulls is
        # False
        if self.name in (BooleanFunction.Name.Any, BooleanFunction.Name.All):
            (ignore_nulls,) = self.options
            (column,) = columns
            is_any = self.name is BooleanFunction.Name.Any
            agg = plc.aggregation.any() if is_any else plc.aggregation.all()
            scalar_result = plc.reduce.reduce(
                column.obj, agg, self.dtype.plc_type, stream=df.stream
            )
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
                h_result = scalar_result.to_py(stream=df.stream)
                if (is_any and not h_result) or (not is_any and h_result):
                    # Any                     All
                    # False || Null => Null   True && Null => Null
                    return Column(
                        plc.Column.all_null_like(column.obj, 1, stream=df.stream),
                        dtype=self.dtype,
                    )
            return Column(
                plc.Column.from_scalar(scalar_result, 1, stream=df.stream),
                dtype=self.dtype,
            )
        if self.name is BooleanFunction.Name.IsNull:
            (column,) = columns
            return Column(
                plc.unary.is_null(column.obj, stream=df.stream), dtype=self.dtype
            )
        elif self.name is BooleanFunction.Name.IsNotNull:
            (column,) = columns
            return Column(
                plc.unary.is_valid(column.obj, stream=df.stream), dtype=self.dtype
            )
        elif self.name in (BooleanFunction.Name.IsNan, BooleanFunction.Name.IsNotNan):
            (column,) = columns
            is_float = column.obj.type().id() in (
                plc.TypeId.FLOAT32,
                plc.TypeId.FLOAT64,
            )
            if is_float:
                op = (
                    plc.unary.is_nan
                    if self.name is BooleanFunction.Name.IsNan
                    else plc.unary.is_not_nan
                )
                base = op(column.obj)
            else:
                base = plc.Column.from_scalar(
                    plc.Scalar.from_py(
                        py_val=self.name is not BooleanFunction.Name.IsNan,
                        stream=df.stream,
                    ),
                    column.size,
                    stream=df.stream,
                )
            out = base.with_mask(column.obj.null_mask(), column.null_count)
            return Column(out, dtype=self.dtype)
        elif self.name is BooleanFunction.Name.IsFirstDistinct:
            (column,) = columns
            return self._distinct(
                column,
                dtype=self.dtype,
                keep=plc.stream_compaction.DuplicateKeepOption.KEEP_FIRST,
                source_value=plc.Scalar.from_py(
                    py_val=True, dtype=self.dtype.plc_type, stream=df.stream
                ),
                target_value=plc.Scalar.from_py(
                    py_val=False, dtype=self.dtype.plc_type, stream=df.stream
                ),
                stream=df.stream,
            )
        elif self.name is BooleanFunction.Name.IsLastDistinct:
            (column,) = columns
            return self._distinct(
                column,
                dtype=self.dtype,
                keep=plc.stream_compaction.DuplicateKeepOption.KEEP_LAST,
                source_value=plc.Scalar.from_py(
                    py_val=True, dtype=self.dtype.plc_type, stream=df.stream
                ),
                target_value=plc.Scalar.from_py(
                    py_val=False,
                    dtype=self.dtype.plc_type,
                    stream=df.stream,
                ),
                stream=df.stream,
            )
        elif self.name is BooleanFunction.Name.IsUnique:
            (column,) = columns
            return self._distinct(
                column,
                dtype=self.dtype,
                keep=plc.stream_compaction.DuplicateKeepOption.KEEP_NONE,
                source_value=plc.Scalar.from_py(
                    py_val=True, dtype=self.dtype.plc_type, stream=df.stream
                ),
                target_value=plc.Scalar.from_py(
                    py_val=False, dtype=self.dtype.plc_type, stream=df.stream
                ),
                stream=df.stream,
            )
        elif self.name is BooleanFunction.Name.IsDuplicated:
            (column,) = columns
            return self._distinct(
                column,
                dtype=self.dtype,
                keep=plc.stream_compaction.DuplicateKeepOption.KEEP_NONE,
                source_value=plc.Scalar.from_py(
                    py_val=False, dtype=self.dtype.plc_type, stream=df.stream
                ),
                target_value=plc.Scalar.from_py(
                    py_val=True, dtype=self.dtype.plc_type, stream=df.stream
                ),
                stream=df.stream,
            )
        elif self.name is BooleanFunction.Name.AllHorizontal:
            return Column(
                reduce(
                    partial(
                        plc.binaryop.binary_operation,
                        op=plc.binaryop.BinaryOperator.NULL_LOGICAL_AND,
                        output_type=self.dtype.plc_type,
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
                        output_type=self.dtype.plc_type,
                    ),
                    (c.obj for c in columns),
                ),
                dtype=self.dtype,
            )
        elif self.name is BooleanFunction.Name.IsIn:
            needles, haystack = columns
            if haystack.obj.type().id() == plc.TypeId.LIST:
                # Unwrap values from the list column
                # .inner returns DataTypeClass | DataType, need to cast to DataType
                haystack = Column(
                    haystack.obj.children()[1],
                    dtype=DataType(
                        cast(
                            pl.DataType, cast(pl.List, haystack.dtype.polars_type).inner
                        )
                    ),
                ).astype(needles.dtype, stream=df.stream)
            if haystack.size:
                return Column(
                    plc.search.contains(
                        haystack.obj,
                        needles.obj,
                        stream=df.stream,
                    ),
                    dtype=self.dtype,
                )
            return Column(
                plc.Column.from_scalar(
                    plc.Scalar.from_py(py_val=False, stream=df.stream),
                    needles.size,
                    stream=df.stream,
                ),
                dtype=self.dtype,
            )
        elif self.name is BooleanFunction.Name.Not:
            (column,) = columns
            # Polars semantics:
            #   integer input: NOT => bitwise invert.
            #   boolean input: NOT => logical NOT.
            return Column(
                plc.unary.unary_operation(
                    column.obj,
                    plc.unary.UnaryOperator.NOT
                    if column.obj.type().id() == plc.TypeId.BOOL8
                    else plc.unary.UnaryOperator.BIT_INVERT,
                    stream=df.stream,
                ),
                dtype=self.dtype,
            )
        else:
            raise NotImplementedError(
                f"BooleanFunction {self.name}"
            )  # pragma: no cover; handled by init raising
