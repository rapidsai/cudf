# Copyright (c) 2024, NVIDIA CORPORATION.

import pyarrow as pa
import pylibcudf as plc
import pytest
from utils import assert_column_eq


@pytest.fixture(scope="module")
def input_data():
    arr1 = ["éâîô\teaio", "ĂĆĖÑÜ", "ACENU", "$24.08", "[a,bb]"]
    arr2 = ["a b", "  c  d\n", "e \t f "]
    return pa.array(arr1), pa.array(arr2)


def test_normalize_spaces(input_data):
    _, input_col = input_data
    result = plc.nvtext.normalize.normalize_spaces(
        plc.interop.from_arrow(input_col)
    )
    expected = pa.array(["a b", "c d", "e f"])
    assert_column_eq(result, expected)


@pytest.mark.parametrize("do_lower", [True, False])
def test_normalize_characters(input_data, do_lower):
    input_col, _ = input_data
    result = plc.nvtext.normalize.normalize_characters(
        plc.interop.from_arrow(input_col),
        do_lower,
    )
    expected = pa.array(
        ["eaio eaio", "acenu", "acenu", " $ 24 . 08", " [ a , bb ] "]
    )
    if not do_lower:
        expected = pa.array(
            ["éâîô eaio", "ĂĆĖÑÜ", "ACENU", " $ 24 . 08", " [ a , bb ] "]
        )
    print(plc.interop.to_arrow(result))
    print(expected)
    assert_column_eq(result, expected)
