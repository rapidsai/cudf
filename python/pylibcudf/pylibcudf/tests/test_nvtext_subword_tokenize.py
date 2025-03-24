# Copyright (c) 2024-2025, NVIDIA CORPORATION.

import pyarrow as pa
import pytest
from utils import assert_column_eq

import pylibcudf as plc


@pytest.mark.parametrize("max_words", [0, 5])
def test_wordpiece_tokenize(max_words):
    vocab_strs = pa.array(["[unk]", "abc", "def", "gh", "##i"])
    vocab = plc.nvtext.wordpiece_tokenize.WordPieceVocabulary(
        plc.interop.from_arrow(vocab_strs)
    )
    input = pa.array(
        [
            "gh def abc xyz ghi defi abci",
            "abc def gh abc def gh abc def gh",
            "abc def gh",
        ]
    )
    result = plc.nvtext.wordpiece_tokenize.wordpiece_tokenize(
        plc.interop.from_arrow(input), vocab, max_words
    )
    pa_result = plc.interop.to_arrow(result)
    if max_words == 5:
        expected = pa.array(
            [[3, 2, 1, 0, 3, 4], [1, 2, 3, 1, 2], [1, 2, 3]],
            type=pa_result.type,
        )
    else:
        expected = pa.array(
            [
                [3, 2, 1, 0, 3, 4, 2, 4, 1, 4],
                [1, 2, 3, 1, 2, 3, 1, 2, 3],
                [1, 2, 3],
            ],
            type=pa_result.type,
        )
    assert_column_eq(result, expected)
