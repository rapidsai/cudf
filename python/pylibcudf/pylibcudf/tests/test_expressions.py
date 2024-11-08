# Copyright (c) 2024, NVIDIA CORPORATION.
import pyarrow as pa
import pyarrow.compute as pc
import pytest
from utils import assert_column_eq

import pylibcudf as plc


def test_literal_construction_invalid():
    with pytest.raises(ValueError):
        plc.expressions.Literal(
            plc.interop.from_arrow(pa.scalar(None, type=pa.list_(pa.int64())))
        )


@pytest.mark.parametrize(
    "tableref",
    [
        plc.expressions.TableReference.LEFT,
        plc.expressions.TableReference.RIGHT,
    ],
)
def test_columnref_construction(tableref):
    plc.expressions.ColumnReference(1, tableref)


def test_columnnameref_construction():
    plc.expressions.ColumnNameReference("abc")


@pytest.mark.parametrize(
    "kwargs",
    [
        # Unary op
        {
            "op": plc.expressions.ASTOperator.IDENTITY,
            "left": plc.expressions.ColumnReference(1),
        },
        # Binop
        {
            "op": plc.expressions.ASTOperator.ADD,
            "left": plc.expressions.ColumnReference(1),
            "right": plc.expressions.ColumnReference(2),
        },
    ],
)
def test_astoperation_construction(kwargs):
    plc.expressions.Operation(**kwargs)


def test_evaluation():
    table_h = pa.table({"a": [1, 2, 3], "b": [4, 5, 6], "c": [7, 8, 9]})
    lit = pa.scalar(42, type=pa.int64())
    table = plc.interop.from_arrow(table_h)
    # expr = abs(b * c - (a + 42))
    expr = plc.expressions.Operation(
        plc.expressions.ASTOperator.ABS,
        plc.expressions.Operation(
            plc.expressions.ASTOperator.SUB,
            plc.expressions.Operation(
                plc.expressions.ASTOperator.MUL,
                plc.expressions.ColumnReference(1),
                plc.expressions.ColumnReference(2),
            ),
            plc.expressions.Operation(
                plc.expressions.ASTOperator.ADD,
                plc.expressions.ColumnReference(0),
                plc.expressions.Literal(plc.interop.from_arrow(lit)),
            ),
        ),
    )

    expect = pc.abs(
        pc.subtract(
            pc.multiply(table_h["b"], table_h["c"]), pc.add(table_h["a"], lit)
        )
    )
    got = plc.transform.compute_column(table, expr)

    assert_column_eq(expect, got)
