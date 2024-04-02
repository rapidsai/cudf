# Copyright (c) 2024, NVIDIA CORPORATION.

import hashlib

import pyarrow as pa
import pytest
from utils import assert_column_eq

import cudf._lib.pylibcudf as plc

SEED = 0
METHODS = ["" "md5", "sha1", "sha224", "sha256", "sha384", "sha512"]


# Full table hash
@pytest.fixture(scope="module")
def all_types_input_table():
    data = pa.Table.from_pydict(
        {
            "int": [1, 2, 3],
            "float": [1.0, 2.0, 3.0],
            "bool": [True, False, True],
            "string": ["a", "b", "c"],
            "list": [[1], [2], [3]],
            #        'struct': [{'a': 1}, {'a': 2}, {'a': 3}]
        }
    )
    return data


def all_types_output_table(input, method):
    def _applyfunc(x):
        hasher = getattr(hashlib, method)
        return hasher(str(x).encode()).hexdigest()

    result = pa.Table.from_pandas(input.to_pandas().map(_applyfunc))
    return result


@pytest.mark.parametrize("method", METHODS)
def test_hash_column(pa_input_column, method):
    def _applyfunc(x):
        hasher = getattr(hashlib, method)
        return hasher(str(x).encode()).hexdigest()

    plc_tbl = plc.interop.from_arrow(
        pa.Table.from_arrays([pa_input_column], names=["data"])
    )
    plc_hasher = getattr(plc.hashing, method)

    expect = pa.Array.from_pandas(
        pa_input_column.to_pandas().apply(_applyfunc)
    )
    got = plc_hasher(plc_tbl)
    assert_column_eq(got, expect)
