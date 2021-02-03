# Copyright (c) 2020, NVIDIA CORPORATION.
from cudf.utils.hash_vocab_utils import hash_vocab
import os
import filecmp
import pytest


@pytest.fixture(scope="module")
def datadir(datadir):
    return os.path.join(datadir, "vocab_hash")


def test_correct_bert_base_vocab_hash(datadir, tmpdir):
    # The vocabulary is 5% drawn from bert-base-uncased
    # sampling script at:
    # https://gist.github.com/VibhuJawa/4fc5981d2cbba1ab8b1e78cdf6aede72
    vocab_path = os.path.join(datadir, "bert-base-uncased-vocab-5per.txt")

    groundtruth_path = os.path.join(
        datadir, "ground_truth_vocab_hash_5per.txt"
    )
    output_path = tmpdir.join("cudf-vocab-hash.txt")
    hash_vocab(vocab_path, output_path)

    assert filecmp.cmp(output_path, groundtruth_path, shallow=False)
