# Copyright (c) 2024, NVIDIA CORPORATION.
import pyarrow as pa
import pylibcudf as plc
import pytest

# We can't really evaluate these expressions, so just make sure
# construction works properly


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
    plc.expressions.ColumnReference(1.0, tableref)


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
