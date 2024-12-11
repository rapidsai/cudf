# Copyright (c) 2024, NVIDIA CORPORATION.

import pyarrow as pa
import pytest
from utils import assert_column_eq

import pylibcudf as plc


@pytest.fixture(scope="module")
def input_data():
    input1 = ["the fuzzy dog", "little piggy", "funny bunny", "chatty parrot"]
    input2 = ["the fuzzy cat", "bitty piggy", "funny bunny", "silent partner"]
    return pa.array(input1), pa.array(input2)


@pytest.mark.parametrize("width", [2, 3])
def test_jaccard_index(input_data, width):
    def get_tokens(s, width):
        return [s[i : i + width] for i in range(len(s) - width + 1)]

    def jaccard_index(s1, s2, width):
        x = set(get_tokens(s1, width))
        y = set(get_tokens(s2, width))
        return len(x & y) / len(x | y)

    input1, input2 = input_data
    result = plc.nvtext.jaccard.jaccard_index(
        plc.interop.from_arrow(input1), plc.interop.from_arrow(input2), width
    )
    expected = pa.array(
        [
            jaccard_index(s1.as_py(), s2.as_py(), width)
            for s1, s2 in zip(input1, input2)
        ],
        type=pa.float32(),
    )
    assert_column_eq(result, expected)
