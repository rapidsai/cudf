# Copyright (c) 2024-2025, NVIDIA CORPORATION.

import pyarrow as pa
import pytest
from utils import assert_column_eq

import pylibcudf as plc


@pytest.fixture(scope="module")
def norm_spaces_input_data():
    arr = ["a b", "  c  d\n", "e \t f "]
    return pa.array(arr)


@pytest.fixture(scope="module")
def norm_chars_input_data():
    arr = ["éâîô\teaio", "ĂĆĖÑÜ", "ACENU", "$24.08", "[a,bb]", "[pad]"]
    return pa.array(arr)


def test_normalize_spaces(norm_spaces_input_data):
    result = plc.nvtext.normalize.normalize_spaces(
        plc.interop.from_arrow(norm_spaces_input_data)
    )
    expected = pa.array(["a b", "c d", "e f"])
    assert_column_eq(result, expected)


@pytest.mark.parametrize("do_lower", [True, False])
def test_normalize_characters(norm_chars_input_data, do_lower):
    result = plc.nvtext.normalize.characters_normalize(
        plc.interop.from_arrow(norm_chars_input_data),
        do_lower,
    )
    if do_lower:
        expected = pa.array(
            [
                "eaio eaio",
                "acenu",
                "acenu",
                " $ 24 . 08",
                " [ a , bb ] ",
                " [ pad ] ",
            ]
        )
    else:
        expected = pa.array(
            [
                "éâîô eaio",
                "ĂĆĖÑÜ",
                "ACENU",
                " $ 24 . 08",
                " [ a , bb ] ",
                " [ pad ] ",
            ]
        )
    assert_column_eq(result, expected)


@pytest.mark.parametrize("do_lower", [True, False])
def test_normalizer(norm_chars_input_data, do_lower):
    result = plc.nvtext.normalize.normalize_characters(
        plc.interop.from_arrow(norm_chars_input_data),
        plc.nvtext.normalize.CharacterNormalizer(
            do_lower,
            plc.column_factories.make_empty_column(plc.types.TypeId.STRING),
        ),
    )
    if do_lower:
        expected = pa.array(
            [
                "eaio eaio",
                "acenu",
                "acenu",
                " $ 24 . 08",
                " [ a , bb ] ",
                " [ pad ] ",
            ]
        )
    else:
        expected = pa.array(
            [
                "éâîô eaio",
                "ĂĆĖÑÜ",
                "ACENU",
                " $ 24 . 08",
                " [ a , bb ] ",
                " [ pad ] ",
            ]
        )
    assert_column_eq(result, expected)


@pytest.mark.parametrize("do_lower", [True, False])
def test_normalizer_with_special_tokens(norm_chars_input_data, do_lower):
    special_tokens = pa.array(["[pad]"])
    result = plc.nvtext.normalize.normalize_characters(
        plc.interop.from_arrow(norm_chars_input_data),
        plc.nvtext.normalize.CharacterNormalizer(
            do_lower, plc.interop.from_arrow(special_tokens)
        ),
    )
    if do_lower:
        expected = pa.array(
            [
                "eaio eaio",
                "acenu",
                "acenu",
                " $ 24 . 08",
                " [ a , bb ] ",
                " [pad] ",
            ]
        )
    else:
        expected = pa.array(
            [
                "éâîô eaio",
                "ĂĆĖÑÜ",
                "ACENU",
                " $ 24 . 08",
                " [ a , bb ] ",
                " [pad] ",
            ]
        )
    assert_column_eq(result, expected)
