# Copyright (c) 2024, NVIDIA CORPORATION.

import pyarrow as pa
import pylibcudf as plc
import pytest
from utils import assert_column_eq


@pytest.fixture(scope="module")
def norm_spaces_input_data():
    arr1 = ["éâîô\teaio", "ĂĆĖÑÜ", "ACENU", "$24.08", "[a,bb]"]
    arr2 = ["a b", "  c  d\n", "e \t f "]
    return pa.array(arr1), pa.array(arr2)


@pytest.fixture(scope="module")
def norm_chars_input_data():
    arr1 = ["éâîô\teaio", "ĂĆĖÑÜ", "ACENU", "$24.08", "[a,bb]"]
    arr2 = ["a b", "  c  d\n", "e \t f "]
    return pa.array(arr1), pa.array(arr2)


def test_normalize_spaces(norm_spaces_input_data):
    result = plc.nvtext.normalize.normalize_spaces(
        plc.interop.from_arrow(norm_spaces_input_data)
    )
    expected = pa.array(["a b", "c d", "e f"])
    assert_column_eq(result, expected)


@pytest.mark.parametrize("do_lower", [True, False])
def test_normalize_characters(norm_chars_input_data, do_lower):
    result = plc.nvtext.normalize.normalize_characters(
        plc.interop.from_arrow(norm_chars_input_data),
        do_lower,
    )
    expected = pa.array(
        ["eaio eaio", "acenu", "acenu", " $ 24 . 08", " [ a , bb ] "]
    )
    if not do_lower:
        expected = pa.array(
            ["éâîô eaio", "ĂĆĖÑÜ", "ACENU", " $ 24 . 08", " [ a , bb ] "]
        )
    assert_column_eq(result, expected)
