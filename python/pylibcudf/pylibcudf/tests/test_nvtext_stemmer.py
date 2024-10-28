# Copyright (c) 2024, NVIDIA CORPORATION.

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

    result = plc.nvtext.stemmer.is_letter(
        plc.interop.from_arrow(input_col),
        check_vowels,
        plc.interop.from_arrow(pa.array(indices))
        if isinstance(indices, list)
        else indices,
    )
    expected = pa.array(
        [
            is_letter(
                s,
                indices[i] if isinstance(indices, list) else indices,
                check_vowels,
            )
            for i, s in enumerate(input_col.to_pylist())
        ]
    )
    assert_column_eq(result, expected)


def test_porter_stemmer_measure(input_col):
    result = plc.nvtext.stemmer.porter_stemmer_measure(
        plc.interop.from_arrow(input_col),
    )
    expected = pa.array([1, 1, 2], type=pa.int32())
    assert_column_eq(result, expected)
