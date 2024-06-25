# Copyright (c) 2024, NVIDIA CORPORATION.
import numpy as np
import pytest

import cudf._lib.pylibcudf as plc

# We can't really evaluate these expressions, so just make sure
# construction works properly


@pytest.mark.parametrize(
    "value", [1, 1.0, "1.0", np.datetime64(1, "ns"), np.timedelta64(1, "ns")]
)
def test_literal_construction(value):
    plc.expressions.Literal(value)


def test_literal_construction_invalid():
    with pytest.raises(NotImplementedError):
        plc.expressions.Literal(object())


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
            "left": plc.expressions.ColumnReference(1.0),
        },
        # Binop
        {
            "op": plc.expressions.ASTOperator.ADD,
            "left": plc.expressions.ColumnReference(1.0),
            "right": plc.expressions.ColumnReference(2.0),
        },
    ],
)
def test_astoperation_construction(kwargs):
    plc.expressions.Operation(**kwargs)
