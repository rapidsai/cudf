# Copyright (c) 2024, NVIDIA CORPORATION.

import pyarrow as pa
import pytest
from utils import assert_column_eq

import pylibcudf as plc


@pytest.fixture(scope="module")
def input_col():
    arr = ["a*b*c*d", "a b c d", "a-b-c-d", "a*b c-d"]
    return pa.array(arr)


@pytest.mark.parametrize("ngrams", [2, 3])
@pytest.mark.parametrize("delim", ["*", " ", "-"])
@pytest.mark.parametrize("sep", ["_", "&", ","])
def test_ngrams_tokenize(input_col, ngrams, delim, sep):
    def ngrams_tokenize(strings, ngrams, delim, sep):
        tokens = []
        for s in strings:
            ss = s.split(delim)
            for i in range(len(ss) - ngrams + 1):
                token = sep.join(ss[i : i + ngrams])
                tokens.append(token)
        return tokens

    result = plc.nvtext.ngrams_tokenize.ngrams_tokenize(
        plc.interop.from_arrow(input_col),
        ngrams,
        plc.interop.from_arrow(pa.scalar(delim)),
        plc.interop.from_arrow(pa.scalar(sep)),
    )
    expected = pa.array(
        ngrams_tokenize(input_col.to_pylist(), ngrams, delim, sep)
    )
    assert_column_eq(result, expected)
