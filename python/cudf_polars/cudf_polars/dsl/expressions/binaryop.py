# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
# TODO: remove need for this
# ruff: noqa: D101
"""BinaryOp DSL nodes."""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

from polars.polars import _expr_nodes as pl_expr

import pylibcudf as plc

from cudf_polars.containers import Column
from cudf_polars.dsl.expressions.base import AggInfo, ExecutionContext, Expr

if TYPE_CHECKING:
    from collections.abc import Mapping

    from cudf_polars.containers import DataFrame

__all__ = ["BinOp"]


class BinOp(Expr):
    __slots__ = ("op",)
    _non_child = ("dtype", "op")

    def __init__(
        self,
        dtype: plc.DataType,
        op: plc.binaryop.BinaryOperator,
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
            self.dtype, left.dtype, right.dtype, op
        ):
            raise NotImplementedError(
                f"Operation {op.name} not supported "
                f"for types {left.dtype.id().name} and {right.dtype.id().name} "
                f"with output type {self.dtype.id().name}"
            )

    _BOOL_KLEENE_MAPPING: ClassVar[
        dict[plc.binaryop.BinaryOperator, plc.binaryop.BinaryOperator]
    ] = {
        plc.binaryop.BinaryOperator.BITWISE_AND: plc.binaryop.BinaryOperator.NULL_LOGICAL_AND,
        plc.binaryop.BinaryOperator.BITWISE_OR: plc.binaryop.BinaryOperator.NULL_LOGICAL_OR,
        plc.binaryop.BinaryOperator.LOGICAL_AND: plc.binaryop.BinaryOperator.NULL_LOGICAL_AND,
        plc.binaryop.BinaryOperator.LOGICAL_OR: plc.binaryop.BinaryOperator.NULL_LOGICAL_OR,
    }

    _MAPPING: ClassVar[dict[pl_expr.Operator, plc.binaryop.BinaryOperator]] = {
        pl_expr.Operator.Eq: plc.binaryop.BinaryOperator.EQUAL,
        pl_expr.Operator.EqValidity: plc.binaryop.BinaryOperator.NULL_EQUALS,
        pl_expr.Operator.NotEq: plc.binaryop.BinaryOperator.NOT_EQUAL,
        pl_expr.Operator.NotEqValidity: plc.binaryop.BinaryOperator.NULL_NOT_EQUALS,
        pl_expr.Operator.Lt: plc.binaryop.BinaryOperator.LESS,
        pl_expr.Operator.LtEq: plc.binaryop.BinaryOperator.LESS_EQUAL,
        pl_expr.Operator.Gt: plc.binaryop.BinaryOperator.GREATER,
        pl_expr.Operator.GtEq: plc.binaryop.BinaryOperator.GREATER_EQUAL,
        pl_expr.Operator.Plus: plc.binaryop.BinaryOperator.ADD,
        pl_expr.Operator.Minus: plc.binaryop.BinaryOperator.SUB,
        pl_expr.Operator.Multiply: plc.binaryop.BinaryOperator.MUL,
        pl_expr.Operator.Divide: plc.binaryop.BinaryOperator.DIV,
        pl_expr.Operator.TrueDivide: plc.binaryop.BinaryOperator.TRUE_DIV,
        pl_expr.Operator.FloorDivide: plc.binaryop.BinaryOperator.FLOOR_DIV,
        pl_expr.Operator.Modulus: plc.binaryop.BinaryOperator.PYMOD,
        pl_expr.Operator.And: plc.binaryop.BinaryOperator.BITWISE_AND,
        pl_expr.Operator.Or: plc.binaryop.BinaryOperator.BITWISE_OR,
        pl_expr.Operator.Xor: plc.binaryop.BinaryOperator.BITWISE_XOR,
        pl_expr.Operator.LogicalAnd: plc.binaryop.BinaryOperator.LOGICAL_AND,
        pl_expr.Operator.LogicalOr: plc.binaryop.BinaryOperator.LOGICAL_OR,
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
            plc.binaryop.binary_operation(lop, rop, self.op, self.dtype),
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
