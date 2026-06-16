# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

import pyarrow as pa
import pytest
from utils import assert_column_eq

import pylibcudf as plc


@pytest.mark.parametrize("max_words", [0, 5])
def test_wordpiece_tokenize(max_words):
    vocab_strs = pa.array(["[unk]", "abc", "def", "gh", "##i"])
    vocab = plc.nvtext.wordpiece_tokenize.WordPieceVocabulary(
        plc.Column.from_arrow(vocab_strs)
    )
    strings_col = pa.array(
        [
            "gh def abc xyz ghi defi abci",
            "abc def gh abc def gh abc def gh",
            "abc def gh",
        ]
    )
    got = plc.nvtext.wordpiece_tokenize.wordpiece_tokenize(
        plc.Column.from_arrow(strings_col), vocab, max_words
    )
    expect_type = got.type().to_arrow(
        value_type=pa.list_(pa.int32()).value_type
    )
    if max_words == 5:
        expect = pa.array(
            [[3, 2, 1, 0, 3, 4], [1, 2, 3, 1, 2], [1, 2, 3]],
            type=expect_type,
        )
    else:
        expect = pa.array(
            [
                [3, 2, 1, 0, 3, 4, 2, 4, 1, 4],
                [1, 2, 3, 1, 2, 3, 1, 2, 3],
                [1, 2, 3],
            ],
            type=expect_type,
        )
    assert_column_eq(expect, got)
