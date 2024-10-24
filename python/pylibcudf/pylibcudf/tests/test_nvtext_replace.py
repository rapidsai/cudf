# Copyright (c) 2024, NVIDIA CORPORATION.

import pyarrow as pa
import pytest
from utils import assert_column_eq

import pylibcudf as plc


@pytest.fixture(scope="module")
def input_col():
    arr = ["the quick", "brown fox", "jumps*over the", "lazy dog"]
    return pa.array(arr)


@pytest.fixture(scope="module")
def targets():
    arr = ["the quick", "brown fox", "jumps*over the", "lazy dog"]
    return pa.array(arr)


@pytest.mark.parametrize("delim", ["*", None])
def test_replace_tokens(input_col, targets, delim):
    replacements = pa.array(["slow", "cat", "looked", "rat"])
    result = plc.nvtext.replace.replace_tokens(
        plc.interop.from_arrow(input_col),
        plc.interop.from_arrow(targets),
        plc.interop.from_arrow(replacements),
        plc.interop.from_arrow(pa.scalar(delim)) if delim else None,
    )
    expected = pa.array(["slow", "cat", "jumps*over the", "rat"])
    if not delim:
        expected = pa.array(
            ["the quick", "brown fox", "jumps*over the", "lazy dog"]
        )
    assert_column_eq(result, expected)


@pytest.mark.parametrize("min_token_length", [4, 5])
@pytest.mark.parametrize("replace", ["---", None])
@pytest.mark.parametrize("delim", ["*", None])
def test_filter_tokens(input_col, min_token_length, replace, delim):
    result = plc.nvtext.replace.filter_tokens(
        plc.interop.from_arrow(input_col),
        min_token_length,
        plc.interop.from_arrow(pa.scalar(replace)) if replace else None,
        plc.interop.from_arrow(pa.scalar(delim)) if delim else None,
    )
    expected = pa.array(
        ["the quick", "brown fox", "jumps*over the", "lazy dog"]
    )
    if not delim and not replace and min_token_length == 4:
        expected = pa.array([" quick", "brown ", "jumps*over ", "lazy "])
    if not delim and not replace and min_token_length == 5:
        expected = pa.array([" quick", "brown ", "jumps*over ", " "])
    if not delim and replace == "---" and min_token_length == 4:
        expected = pa.array(
            ["--- quick", "brown ---", "jumps*over ---", "lazy ---"]
        )
    if not delim and replace == "---" and min_token_length == 5:
        expected = pa.array(
            ["--- quick", "brown ---", "jumps*over ---", "--- ---"]
        )
    assert_column_eq(result, expected)
