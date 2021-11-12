# Copyright (c) 2020-2021, NVIDIA CORPORATION.
import os

import numpy as np
import pytest
from transformers import BertTokenizer

import cudf
from cudf.core.subword_tokenizer import SubwordTokenizer


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


def test_subword_tokenize_on_disk_vocab_str_api(datadir):
    """
    Tests the subword-tokenizer API where
    the vocabulary is not pre-loaded
    and is accessed via the string accessor
    """
    with open(
        os.path.join(datadir, "test_sentences.txt"), encoding="utf-8"
    ) as file:
        input_sentence_ls = [line.strip() for line in file]

    vocab_dir = os.path.join(datadir, "bert_base_cased_sampled")
    vocab_hash_path = os.path.join(vocab_dir, "vocab-hash.txt")

    ser = cudf.Series(input_sentence_ls)
    tokens, masks, metadata = ser.str.subword_tokenize(
        vocab_hash_path,
        max_length=32,
        stride=32,
        do_lower=True,
        max_rows_tensor=len(ser),
    )


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

    hf_tokenizer = BertTokenizer.from_pretrained(
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
