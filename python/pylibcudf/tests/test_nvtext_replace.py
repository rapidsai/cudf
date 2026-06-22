# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

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
    got = plc.nvtext.replace.replace_tokens(
        plc.Column.from_arrow(input_col),
        plc.Column.from_arrow(targets),
        plc.Column.from_arrow(replacements),
        plc.Scalar.from_arrow(pa.scalar(delim)) if delim else None,
    )
    expect = pa.array(["slow", "cat", "jumps*over the", "rat"])
    if not delim:
        expect = pa.array(
            ["the quick", "brown fox", "jumps*over the", "lazy dog"]
        )
    assert_column_eq(expect, got)


@pytest.mark.parametrize("min_token_length", [4, 5])
@pytest.mark.parametrize("replace", ["---", None])
@pytest.mark.parametrize("delim", ["*", None])
def test_filter_tokens(input_col, min_token_length, replace, delim):
    got = plc.nvtext.replace.filter_tokens(
        plc.Column.from_arrow(input_col),
        min_token_length,
        plc.Scalar.from_arrow(pa.scalar(replace)) if replace else None,
        plc.Scalar.from_arrow(pa.scalar(delim)) if delim else None,
    )
    expect = pa.array(["the quick", "brown fox", "jumps*over the", "lazy dog"])
    if not delim and not replace and min_token_length == 4:
        expect = pa.array([" quick", "brown ", "jumps*over ", "lazy "])
    if not delim and not replace and min_token_length == 5:
        expect = pa.array([" quick", "brown ", "jumps*over ", " "])
    if not delim and replace == "---" and min_token_length == 4:
        expect = pa.array(
            ["--- quick", "brown ---", "jumps*over ---", "lazy ---"]
        )
    if not delim and replace == "---" and min_token_length == 5:
        expect = pa.array(
            ["--- quick", "brown ---", "jumps*over ---", "--- ---"]
        )
    assert_column_eq(expect, got)
