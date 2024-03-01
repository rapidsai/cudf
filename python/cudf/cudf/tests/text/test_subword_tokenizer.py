# Copyright (c) 2020-2024, NVIDIA CORPORATION.
import os

import cupy
import numpy as np
import pytest

import cudf
from cudf.core.subword_tokenizer import SubwordTokenizer
from cudf.testing._utils import assert_eq


@pytest.fixture(scope="module")
def datadir(datadir):
    return os.path.join(datadir, "subword_tokenizer_data")


def assert_equal_tokenization_outputs(hf_output, cudf_output):
    assert (
        np.sum(hf_output["input_ids"] != cudf_output["input_ids"].get()) == 0
    )
    assert (
        np.sum(
            hf_output["attention_mask"] != cudf_output["attention_mask"].get()
        )
        == 0
    )


@pytest.mark.skip(reason="segfaults")
@pytest.mark.parametrize("seq_len", [32, 64])
@pytest.mark.parametrize("stride", [0, 15, 30])
@pytest.mark.parametrize("add_special_tokens", [True, False])
@pytest.mark.parametrize("do_lower_case", [True, False])
def test_subword_tokenize(
    seq_len, stride, add_special_tokens, do_lower_case, datadir
):
    with open(
        os.path.join(datadir, "test_sentences.txt"), encoding="utf-8"
    ) as file:
        input_sentence_ls = [line.strip() for line in file]

    vocab_dir = os.path.join(datadir, "bert_base_cased_sampled")

    transformers = pytest.importorskip("transformers")

    hf_tokenizer = transformers.BertTokenizer.from_pretrained(
        vocab_dir, do_lower_case=do_lower_case
    )

    hf_output = hf_tokenizer(
        input_sentence_ls,
        max_length=seq_len,
        stride=stride,
        padding="max_length",
        return_tensors="np",
        truncation=True,
        add_special_tokens=add_special_tokens,
    )

    vocab_hash = os.path.join(vocab_dir, "vocab-hash.txt")
    str_series = cudf.Series(input_sentence_ls)
    cudf_tokenizer = SubwordTokenizer(vocab_hash, do_lower_case=do_lower_case)
    cudf_output = cudf_tokenizer(
        str_series,
        max_length=seq_len,
        max_num_rows=len(str_series),
        stride=stride,
        padding="max_length",
        return_tensors="cp",
        truncation=True,
        add_special_tokens=add_special_tokens,
    )
    assert_equal_tokenization_outputs(hf_output, cudf_output)


def test_subword_tokenize_with_truncation(datadir):
    vocab_dir = os.path.join(datadir, "bert_base_cased_sampled")
    vocab_hash = os.path.join(vocab_dir, "vocab-hash.txt")
    str_series = cudf.Series(["Test error"])
    cudf_tokenizer = SubwordTokenizer(vocab_hash)

    error_msg = (
        "Adding special tokens is not supported with truncation = False. "
        "Custom Cupy kernel can potentially "
        "be used to add it. For reference "
        "see: _bert_add_special_tokens"
    )

    with pytest.raises(NotImplementedError, match=error_msg):
        cudf_tokenizer(
            str_series,
            max_length=64,
            max_num_rows=len(str_series),
            truncation=False,
            add_special_tokens=True,
        )


def test_text_subword_tokenize(tmpdir):
    sr = cudf.Series(
        [
            "This is a test",
            "A test this is",
            "Is test a this",
            "Test   test",
            "this   This",
        ]
    )
    hash_file = tmpdir.mkdir("nvtext").join("tmp_hashed_vocab.txt")
    content = "1\n0\n23\n"
    coefficients = [65559] * 23
    for c in coefficients:
        content = content + str(c) + " 0\n"
    # based on values from the bert_hash_table.txt file for the
    # test words used here: 'this' 'is' 'a' test'
    table = [0] * 23
    table[0] = 3015668
    table[1] = 6205475701751155871
    table[5] = 6358029
    table[16] = 451412625363
    table[20] = 6206321707968235495
    content = content + "23\n"
    for v in table:
        content = content + str(v) + "\n"
    content = content + "100\n101\n102\n\n"
    hash_file.write(content)

    cudf_tokenizer = SubwordTokenizer(hash_file)

    token_d = cudf_tokenizer(
        sr, 8, 8, add_special_tokens=False, truncation=True
    )
    tokens, masks, metadata = (
        token_d["input_ids"],
        token_d["attention_mask"],
        token_d["metadata"],
    )
    expected_tokens = cupy.asarray(
        [
            2023,
            2003,
            1037,
            3231,
            0,
            0,
            0,
            0,
            1037,
            3231,
            2023,
            2003,
            0,
            0,
            0,
            0,
            2003,
            3231,
            1037,
            2023,
            0,
            0,
            0,
            0,
            3231,
            3231,
            0,
            0,
            0,
            0,
            0,
            0,
            2023,
            2023,
            0,
            0,
            0,
            0,
            0,
            0,
        ],
        dtype=np.uint32,
    )
    expected_tokens = expected_tokens.reshape(-1, 8)
    assert_eq(expected_tokens, tokens)

    expected_masks = cupy.asarray(
        [
            1,
            1,
            1,
            1,
            0,
            0,
            0,
            0,
            1,
            1,
            1,
            1,
            0,
            0,
            0,
            0,
            1,
            1,
            1,
            1,
            0,
            0,
            0,
            0,
            1,
            1,
            0,
            0,
            0,
            0,
            0,
            0,
            1,
            1,
            0,
            0,
            0,
            0,
            0,
            0,
        ],
        dtype=np.uint32,
    )
    expected_masks = expected_masks.reshape(-1, 8)
    assert_eq(expected_masks, masks)

    expected_metadata = cupy.asarray(
        [0, 0, 3, 1, 0, 3, 2, 0, 3, 3, 0, 1, 4, 0, 1], dtype=np.uint32
    )
    expected_metadata = expected_metadata.reshape(-1, 3)
    assert_eq(expected_metadata, metadata)
