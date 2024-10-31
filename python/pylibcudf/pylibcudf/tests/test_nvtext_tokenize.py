# Copyright (c) 2024, NVIDIA CORPORATION.

import pyarrow as pa
import pytest
from utils import assert_column_eq

import pylibcudf as plc


@pytest.fixture(scope="module")
def input_col():
    return pa.array(["a", "b c", "d.e:f;"])


@pytest.mark.parametrize(
    "delimiter", [None, plc.interop.from_arrow(pa.scalar("."))]
)
def test_tokenize_scalar(input_col, delimiter):
    result = plc.nvtext.tokenize.tokenize_scalar(
        plc.interop.from_arrow(input_col), delimiter
    )
    if delimiter is None:
        expected = pa.array(["a", "b", "c", "d.e:f;"])
    else:
        expected = pa.array(["a", "b c", "d", "e:f;"])
    assert_column_eq(result, expected)


def test_tokenize_column(input_col):
    delimiters = pa.array([" ", ".", ":", ";"])
    result = plc.nvtext.tokenize.tokenize_column(
        plc.interop.from_arrow(input_col), plc.interop.from_arrow(delimiters)
    )
    expected = pa.array(["a", "b", "c", "d", "e", "f"])
    assert_column_eq(result, expected)


@pytest.mark.parametrize(
    "delimiter", [None, plc.interop.from_arrow(pa.scalar("."))]
)
def test_count_tokens_scalar(input_col, delimiter):
    result = plc.nvtext.tokenize.count_tokens_scalar(
        plc.interop.from_arrow(input_col), delimiter
    )
    if delimiter is None:
        expected = pa.array([1, 2, 1], type=pa.int32())
    else:
        expected = pa.array([1, 1, 2], type=pa.int32())
    assert_column_eq(result, expected)


def test_count_tokens_column(input_col):
    delimiters = pa.array([" ", ".", ":", ";"])
    result = plc.nvtext.tokenize.count_tokens_column(
        plc.interop.from_arrow(input_col), plc.interop.from_arrow(delimiters)
    )
    expected = pa.array([1, 2, 3], type=pa.int32())
    assert_column_eq(result, expected)


def test_character_tokenize(input_col):
    result = plc.nvtext.tokenize.character_tokenize(
        plc.interop.from_arrow(input_col)
    )
    expected = pa.array(["a", "b", " ", "c", "d", ".", "e", ":", "f", ";"])
    assert_column_eq(result, expected)


@pytest.mark.parametrize(
    "delimiter", [None, plc.interop.from_arrow(pa.scalar("."))]
)
def test_detokenize(input_col, delimiter):
    row_indices = pa.array([0, 0, 1])
    result = plc.nvtext.tokenize.detokenize(
        plc.interop.from_arrow(input_col), plc.interop.from_arrow(row_indices)
    )
    expected = pa.array(["a b c", "d.e:f;"])
    assert_column_eq(result, expected)


@pytest.mark.parametrize("default_id", [-1, 0])
def test_tokenize_with_vocabulary(input_col, default_id):
    result = plc.nvtext.tokenize.tokenize_with_vocabulary(
        plc.interop.from_arrow(input_col),
        plc.nvtext.tokenize.TokenizeVocabulary(
            plc.interop.from_arrow(input_col)
        ),
        plc.interop.from_arrow(pa.scalar(" ")),
        default_id,
    )
    pa_result = plc.interop.to_arrow(result)
    if default_id == -1:
        expected = pa.array([[0], [-1, -1], [2]], type=pa_result.type)
    else:
        expected = pa.array([[0], [0, 0], [2]], type=pa_result.type)
    assert_column_eq(result, expected)
