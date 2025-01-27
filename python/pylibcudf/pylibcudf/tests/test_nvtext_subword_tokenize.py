# Copyright (c) 2024, NVIDIA CORPORATION.

import pyarrow as pa
import pytest
from utils import assert_column_eq

import pylibcudf as plc


@pytest.fixture
def vocab_file(tmpdir):
    hash_file = tmpdir.mkdir("nvtext").join("tmp_hashed_vocab.txt")
    content = "1\n0\n10\n"
    coefficients = [65559] * 10
    for c in coefficients:
        content = content + str(c) + " 0\n"
    table = [0] * 10
    table[0] = 3015668
    content = content + "10\n"
    for v in table:
        content = content + str(v) + "\n"
    content = content + "100\n101\n102\n\n"
    hash_file.write(content)
    return str(hash_file)


@pytest.fixture
def column_input():
    return pa.array(["This is a test"])


@pytest.mark.parametrize("max_sequence_length", [64, 128])
@pytest.mark.parametrize("stride", [32, 64])
@pytest.mark.parametrize("do_lower_case", [True, False])
@pytest.mark.parametrize("do_truncate", [True, False])
def test_subword_tokenize(
    vocab_file,
    column_input,
    max_sequence_length,
    stride,
    do_lower_case,
    do_truncate,
):
    vocab = plc.nvtext.subword_tokenize.HashedVocabulary(vocab_file)
    tokens, masks, metadata = plc.nvtext.subword_tokenize.subword_tokenize(
        plc.interop.from_arrow(column_input),
        vocab,
        max_sequence_length,
        stride,
        do_lower_case,
        do_truncate,
    )
    expected_tokens = pa.array(
        [100] * 4 + [0] * (max_sequence_length - 4), type=pa.uint32()
    )
    expected_masks = pa.array(
        [1] * 4 + [0] * (max_sequence_length - 4), type=pa.uint32()
    )
    expected_metadata = pa.array([0, 0, 3], type=pa.uint32())

    assert_column_eq(tokens, expected_tokens)
    assert_column_eq(masks, expected_masks)
    assert_column_eq(metadata, expected_metadata)
