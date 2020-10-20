# Copyright (c) 2020, NVIDIA CORPORATION.
from cudf.utils.hash_vocab_utils import hash_vocab
import os
import filecmp
import pytest


@pytest.fixture(scope="module")
def datadir(datadir):
    return os.path.join(datadir, "vocab_hash")


def test_correct_bert_base_vocab_hash(datadir, tmpdir):
    vocab_path = os.path.join(datadir, "bert-base-uncased-vocab.txt")
    groundtruth_path = os.path.join(datadir, "ground_truth_vocab_hash.txt")
    output_path = tmpdir.join("cudf-vocab-hash.txt")
    hash_vocab(vocab_path, output_path)

    assert filecmp.cmp(output_path, groundtruth_path, shallow=False)
