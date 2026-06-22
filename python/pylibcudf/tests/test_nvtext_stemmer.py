# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

import pyarrow as pa
import pytest
from utils import assert_column_eq

import pylibcudf as plc


@pytest.fixture(scope="module")
def input_col():
    arr = ["trouble", "toy", "syzygy"]
    return pa.array(arr)


@pytest.mark.parametrize("check_vowels", [True, False])
@pytest.mark.parametrize("indices", [[3, 1, 4], 1])
def test_is_letter(input_col, check_vowels, indices):
    def is_letter(s, i, check):
        vowels = "aeiouy"
        return (s[i] in vowels) == check

    got = plc.nvtext.stemmer.is_letter(
        plc.Column.from_arrow(input_col),
        check_vowels,
        plc.Column.from_arrow(pa.array(indices))
        if isinstance(indices, list)
        else indices,
    )
    expect = pa.array(
        [
            is_letter(
                s,
                indices[i] if isinstance(indices, list) else indices,
                check_vowels,
            )
            for i, s in enumerate(input_col.to_pylist())
        ]
    )
    assert_column_eq(expect, got)


def test_porter_stemmer_measure(input_col):
    got = plc.nvtext.stemmer.porter_stemmer_measure(
        plc.Column.from_arrow(input_col),
    )
    expect = pa.array([1, 1, 2], type=pa.int32())
    assert_column_eq(expect, got)
