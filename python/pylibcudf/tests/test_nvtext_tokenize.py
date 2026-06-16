# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

import pyarrow as pa
import pytest
from utils import assert_column_eq

import pylibcudf as plc


@pytest.fixture(scope="module")
def input_col():
    return pa.array(["a", "b c", "d.e:f;"])


@pytest.mark.parametrize(
    "delimiter", [None, plc.Scalar.from_arrow(pa.scalar("."))]
)
def test_tokenize_scalar(input_col, delimiter):
    got = plc.nvtext.tokenize.tokenize_scalar(
        plc.Column.from_arrow(input_col), delimiter
    )
    if delimiter is None:
        expect = pa.array(["a", "b", "c", "d.e:f;"])
    else:
        expect = pa.array(["a", "b c", "d", "e:f;"])
    assert_column_eq(expect, got)


def test_tokenize_column(input_col):
    got = plc.nvtext.tokenize.tokenize_column(
        plc.Column.from_arrow(input_col),
        plc.Column.from_arrow(pa.array([" ", ".", ":", ";"])),
    )
    expect = pa.array(["a", "b", "c", "d", "e", "f"])
    assert_column_eq(expect, got)


@pytest.mark.parametrize(
    "delimiter", [None, plc.Scalar.from_arrow(pa.scalar("."))]
)
def test_count_tokens_scalar(input_col, delimiter):
    got = plc.nvtext.tokenize.count_tokens_scalar(
        plc.Column.from_arrow(input_col), delimiter
    )
    if delimiter is None:
        expect = pa.array([1, 2, 1], type=pa.int32())
    else:
        expect = pa.array([1, 1, 2], type=pa.int32())
    assert_column_eq(expect, got)


def test_count_tokens_column(input_col):
    got = plc.nvtext.tokenize.count_tokens_column(
        plc.Column.from_arrow(input_col),
        plc.Column.from_arrow(pa.array([" ", ".", ":", ";"])),
    )
    expect = pa.array([1, 2, 3], type=pa.int32())
    assert_column_eq(expect, got)


def test_character_tokenize(input_col):
    got = plc.nvtext.tokenize.character_tokenize(
        plc.Column.from_arrow(input_col)
    )
    expect = pa.array([["a"], ["b", " ", "c"], ["d", ".", "e", ":", "f", ";"]])
    assert_column_eq(expect, got)


@pytest.mark.parametrize(
    "delimiter", [None, plc.Scalar.from_arrow(pa.scalar("."))]
)
def test_detokenize(input_col, delimiter):
    row_indices = pa.array([0, 0, 1])
    got = plc.nvtext.tokenize.detokenize(
        plc.Column.from_arrow(input_col), plc.Column.from_arrow(row_indices)
    )
    expect = pa.array(["a b c", "d.e:f;"])
    assert_column_eq(expect, got)


@pytest.mark.parametrize("default_id", [-1, 0])
def test_tokenize_with_vocabulary(input_col, default_id):
    got = plc.nvtext.tokenize.tokenize_with_vocabulary(
        plc.Column.from_arrow(input_col),
        plc.nvtext.tokenize.TokenizeVocabulary(
            plc.Column.from_arrow(input_col)
        ),
        plc.Scalar.from_arrow(pa.scalar(" ")),
        default_id,
    )
    expect_type = got.type().to_arrow(
        value_type=pa.list_(pa.int32()).value_type
    )
    if default_id == -1:
        expect = pa.array([[0], [-1, -1], [2]], type=expect_type)
    else:
        expect = pa.array([[0], [0, 0], [2]], type=expect_type)
    assert_column_eq(expect, got)
