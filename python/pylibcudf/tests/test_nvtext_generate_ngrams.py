# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

import pyarrow as pa
import pytest
from utils import assert_column_eq

import pylibcudf as plc


@pytest.fixture(scope="module")
def input_col():
    arr = ["ab", "cde", "fgh"]
    return pa.array(arr)


@pytest.mark.parametrize("ngram", [2, 3])
@pytest.mark.parametrize("sep", ["_", "**", ","])
def test_generate_ngrams(input_col, ngram, sep):
    got = plc.nvtext.generate_ngrams.generate_ngrams(
        plc.Column.from_arrow(input_col),
        ngram,
        plc.Scalar.from_arrow(pa.scalar(sep)),
    )
    expect = pa.array([f"ab{sep}cde", f"cde{sep}fgh"])
    if ngram == 3:
        expect = pa.array([f"ab{sep}cde{sep}fgh"])
    assert_column_eq(expect, got)


@pytest.mark.parametrize("ngram", [2, 3])
def test_generate_character_ngrams(input_col, ngram):
    got = plc.nvtext.generate_ngrams.generate_character_ngrams(
        plc.Column.from_arrow(input_col),
        ngram,
    )
    expect = pa.array([["ab"], ["cd", "de"], ["fg", "gh"]])
    if ngram == 3:
        expect = pa.array([[], ["cde"], ["fgh"]])
    assert_column_eq(expect, got)


@pytest.mark.parametrize("ngram", [2, 3])
@pytest.mark.parametrize("seed", [0, 3])
def test_hash_character_ngrams(input_col, ngram, seed):
    pa_result = plc.nvtext.generate_ngrams.hash_character_ngrams(
        plc.Column.from_arrow(input_col), ngram, seed
    ).to_arrow()
    assert all(
        len(got) == max(0, len(s.as_py()) - ngram + 1)
        for got, s in zip(pa_result, input_col, strict=True)
    )
    assert pa_result.type == pa.list_(
        pa.field("element", pa.uint32(), nullable=False)
    )
