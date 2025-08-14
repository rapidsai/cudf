# Copyright (c) 2025, NVIDIA CORPORATION.
import hashlib

import pytest

import cudf
from cudf.testing import assert_eq


@pytest.mark.parametrize(
    "method", ["md5", "sha1", "sha224", "sha256", "sha384", "sha512"]
)
def test_series_hash_values(method):
    inputs = cudf.Series(
        [
            "",
            "0",
            "A 56 character string to test message padding algorithm.",
            "A 63 character string to test message padding algorithm, again.",
            "A 64 character string to test message padding algorithm, again!!",
            (
                "A very long (greater than 128 bytes/char string) to execute "
                "a multi hash-step data point in the hash function being "
                "tested. This string needed to be longer."
            ),
            "All work and no play makes Jack a dull boy",
            "!\"#$%&'()*+,-./0123456789:;<=>?@[\\]^_`{|}~",
            "\x00\x00\x00\x10\x00\x00\x00\x00",
            "\x00\x00\x00\x00",
        ]
    )

    def hashlib_compute_digest(data):
        hasher = getattr(hashlib, method)()
        hasher.update(data.encode("utf-8"))
        return hasher.hexdigest()

    hashlib_validation = inputs.to_pandas().apply(hashlib_compute_digest)
    validation_results = cudf.Series(hashlib_validation)
    hash_values = inputs.hash_values(method=method)
    assert_eq(hash_values, validation_results)


def test_series_hash_values_invalid_method():
    inputs = cudf.Series(["", "0"])
    with pytest.raises(ValueError):
        inputs.hash_values(method="invalid_method")
