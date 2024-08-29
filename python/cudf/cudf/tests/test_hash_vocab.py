# Copyright (c) 2020-2024, NVIDIA CORPORATION.
import filecmp
import os
import warnings

import pytest

from cudf.utils.hash_vocab_utils import hash_vocab


@pytest.fixture(scope="module")
def datadir(datadir):
    return os.path.join(
        datadir, "subword_tokenizer_data", "bert_base_cased_sampled"
    )


def test_correct_bert_base_vocab_hash(datadir, tmpdir):
    # The vocabulary is drawn from bert-base-cased
    vocab_path = os.path.join(datadir, "vocab.txt")

    groundtruth_path = os.path.join(datadir, "vocab-hash.txt")
    output_path = tmpdir.join("cudf-vocab-hash.txt")
    warnings.simplefilter(action="ignore", category=RuntimeWarning)
    hash_vocab(vocab_path, output_path)

    assert filecmp.cmp(output_path, groundtruth_path, shallow=False)
