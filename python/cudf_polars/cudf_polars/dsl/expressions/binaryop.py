# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
# TODO: remove need for this
# ruff: noqa: D101
"""BinaryOp DSL nodes."""

from __future__ import annotations

from enum import IntEnum, auto
from typing import TYPE_CHECKING, ClassVar

from polars.polars import _expr_nodes as pl_expr

import pylibcudf as plc
from pylibcudf import expressions as plc_expr

from cudf_polars.containers import Column
from cudf_polars.dsl.expressions.base import AggInfo, ExecutionContext, Expr

if TYPE_CHECKING:
    from collections.abc import Mapping

    from typing_extensions import Self

    from cudf_polars.containers import DataFrame

__all__ = ["BinOp"]


class BinOp(Expr):
    __slots__ = ("op",)
    _non_child = ("dtype", "op")

    class Operator(IntEnum):
        """Internal and picklable representation of pylibcudf's `BinaryOperator`."""

        ADD = auto()
        ATAN2 = auto()
        BITWISE_AND = auto()
        BITWISE_OR = auto()
        BITWISE_XOR = auto()
        DIV = auto()
        EQUAL = auto()
        FLOOR_DIV = auto()
        GENERIC_BINARY = auto()
        GREATER = auto()
        GREATER_EQUAL = auto()
        INT_POW = auto()
        INVALID_BINARY = auto()
        LESS = auto()
        LESS_EQUAL = auto()
        LOGICAL_AND = auto()
        LOGICAL_OR = auto()
        LOG_BASE = auto()
        MOD = auto()
        MUL = auto()
        NOT_EQUAL = auto()
        NULL_EQUALS = auto()
        NULL_LOGICAL_AND = auto()
        NULL_LOGICAL_OR = auto()
        NULL_MAX = auto()
        NULL_MIN = auto()
        NULL_NOT_EQUALS = auto()
        PMOD = auto()
        POW = auto()
        PYMOD = auto()
        SHIFT_LEFT = auto()
        SHIFT_RIGHT = auto()
        SHIFT_RIGHT_UNSIGNED = auto()
        SUB = auto()
        TRUE_DIV = auto()

        @classmethod
        def from_polars(cls, obj: pl_expr.Operator) -> BinOp.Operator:
            """Convert from polars' `Operator`."""
            mapping: dict[pl_expr.Operator, BinOp.Operator] = {
                pl_expr.Operator.Eq: BinOp.Operator.EQUAL,
                pl_expr.Operator.EqValidity: BinOp.Operator.NULL_EQUALS,
                pl_expr.Operator.NotEq: BinOp.Operator.NOT_EQUAL,
                pl_expr.Operator.NotEqValidity: BinOp.Operator.NULL_NOT_EQUALS,
                pl_expr.Operator.Lt: BinOp.Operator.LESS,
                pl_expr.Operator.LtEq: BinOp.Operator.LESS_EQUAL,
                pl_expr.Operator.Gt: BinOp.Operator.GREATER,
                pl_expr.Operator.GtEq: BinOp.Operator.GREATER_EQUAL,
                pl_expr.Operator.Plus: BinOp.Operator.ADD,
                pl_expr.Operator.Minus: BinOp.Operator.SUB,
                pl_expr.Operator.Multiply: BinOp.Operator.MUL,
                pl_expr.Operator.Divide: BinOp.Operator.DIV,
                pl_expr.Operator.TrueDivide: BinOp.Operator.TRUE_DIV,
                pl_expr.Operator.FloorDivide: BinOp.Operator.FLOOR_DIV,
                pl_expr.Operator.Modulus: BinOp.Operator.PYMOD,
                pl_expr.Operator.And: BinOp.Operator.BITWISE_AND,
                pl_expr.Operator.Or: BinOp.Operator.BITWISE_OR,
                pl_expr.Operator.Xor: BinOp.Operator.BITWISE_XOR,
                pl_expr.Operator.LogicalAnd: BinOp.Operator.LOGICAL_AND,
                pl_expr.Operator.LogicalOr: BinOp.Operator.LOGICAL_OR,
            }

            return mapping[obj]

        @classmethod
        def to_pylibcudf(cls, obj: Self) -> plc.binaryop.BinaryOperator:
            """Convert to pylibcudf's `BinaryOperator`."""
            return getattr(plc.binaryop.BinaryOperator, obj.name)

        @classmethod
        def to_pylibcudf_expr(cls, obj: Self) -> plc.binaryop.BinaryOperator:
            """Convert to pylibcudf's `ASTOperator`."""
            if obj is BinOp.Operator.NULL_EQUALS:
                # Name mismatch in pylibcudf's `BinaryOperator` and `ASTOperator`.
                return plc_expr.ASTOperator.NULL_EQUAL
            return getattr(plc_expr.ASTOperator, obj.name)

    def __init__(
        self,
        dtype: plc.DataType,
        op: BinOp.Operator,
        left: Expr,
        right: Expr,
    ) -> None:
        self.dtype = dtype
        if plc.traits.is_boolean(self.dtype):
            # For boolean output types, bitand and bitor implement
            # boolean logic, so translate. bitxor also does, but the
            # default behaviour is correct.
            op = BinOp._BOOL_KLEENE_MAPPING.get(op, op)
        self.op = op
        self.children = (left, right)
        if not plc.binaryop.is_supported_operation(
            self.dtype, left.dtype, right.dtype, BinOp.Operator.to_pylibcudf(op)
        ):
            raise NotImplementedError(
                f"Operation {op.name} not supported "
                f"for types {left.dtype.id().name} and {right.dtype.id().name} "
                f"with output type {self.dtype.id().name}"
            )

    _BOOL_KLEENE_MAPPING: ClassVar[dict[Operator, Operator]] = {
        Operator.BITWISE_AND: Operator.NULL_LOGICAL_AND,
        Operator.BITWISE_OR: Operator.NULL_LOGICAL_OR,
        Operator.LOGICAL_AND: Operator.NULL_LOGICAL_AND,
        Operator.LOGICAL_OR: Operator.NULL_LOGICAL_OR,
    }

    def do_evaluate(
        self,
        df: DataFrame,
        *,
        context: ExecutionContext = ExecutionContext.FRAME,
        mapping: Mapping[Expr, Column] | None = None,
    ) -> Column:
        """Evaluate this expression given a dataframe for context."""
        left, right = (
            child.evaluate(df, context=context, mapping=mapping)
            for child in self.children
        )
        lop = left.obj
        rop = right.obj
        if left.obj.size() != right.obj.size():
            if left.is_scalar:
                lop = left.obj_scalar
            elif right.is_scalar:
                rop = right.obj_scalar
        return Column(
            plc.binaryop.binary_operation(
                lop, rop, BinOp.Operator.to_pylibcudf(self.op), self.dtype
            ),
        )

    def collect_agg(self, *, depth: int) -> AggInfo:
        """Collect information about aggregations in groupbys."""
        if depth == 1:
            # inside aggregation, need to pre-evaluate,
            # groupby construction has checked that we don't have
            # nested aggs, so stop the recursion and return ourselves
            # for pre-eval
            return AggInfo([(self, plc.aggregation.collect_list(), self)])
        else:
            left_info, right_info = (
                child.collect_agg(depth=depth) for child in self.children
            )
            requests = [*left_info.requests, *right_info.requests]
            # TODO: Hack, if there were no reductions inside this
            # binary expression then we want to pre-evaluate and
            # collect ourselves. Otherwise we want to collect the
            # aggregations inside and post-evaluate. This is a bad way
            # of checking that we are in case 1.
            if all(
                agg.kind() == plc.aggregation.Kind.COLLECT_LIST
                for _, agg, _ in requests
            ):
                return AggInfo([(self, plc.aggregation.collect_list(), self)])
            return AggInfo(
                [*left_info.requests, *right_info.requests],
            )
