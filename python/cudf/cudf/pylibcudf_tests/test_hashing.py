# Copyright (c) 2024, NVIDIA CORPORATION.

import hashlib
import struct

import numpy as np
import pyarrow as pa
import pytest
import xxhash
from utils import assert_column_eq, assert_table_eq

import cudf._lib.pylibcudf as plc

SEED = 0
METHODS = ["md5", "sha1", "sha224", "sha256", "sha384", "sha512"]


@pytest.fixture(scope="module")
def xxhash_64_int_tbl():
    arrow_tbl = pa.Table.from_arrays(
        [
            pa.array(
                [
                    -127,
                    -70000,
                    0,
                    200000,
                    128,
                    np.iinfo("int32").max,
                    np.iinfo("int32").min,
                    np.iinfo("int32").min,
                ],
                type=pa.int32(),
            )
        ],
        names=["data"],
    )
    return plc.interop.from_arrow(arrow_tbl)


@pytest.fixture(scope="module")
def xxhash_64_double_tbl():
    arrow_tbl = pa.Table.from_arrays(
        [
            pa.array(
                [
                    -127.0,
                    -70000.125,
                    128.5,
                    -0.0,
                    np.inf,
                    np.nan,
                    np.finfo("float64").max,
                    np.finfo("float64").min,
                    np.finfo("float64").min,
                ],
                type=pa.float32(),
            )
        ],
        names=["data"],
    )
    return plc.interop.from_arrow(arrow_tbl)


@pytest.fixture(scope="module")
def xxhash_64_string_tbl():
    arrow_tbl = pa.Table.from_arrays(
        [
            pa.array(
                [
                    "The",
                    "quick",
                    "brown fox",
                    "jumps over the lazy dog.",
                    "I am Jack's complete lack of null value",
                    "A very long (greater than 128 bytes/characters) to test a very long string. "
                    "2nd half of the very long string to verify the long string hashing happening.",
                    "Some multi-byte characters here: ééé",
                    "ééé",
                    "ééé ééé",
                    "ééé ééé ééé ééé",
                    "",
                    "!@#$%^&*(())",
                    "0123456789",
                    "{}|:<>?,./;[]=-",
                ],
                type=pa.string(),
            )
        ],
        names=["data"],
    )
    return plc.interop.from_arrow(arrow_tbl)


@pytest.fixture(scope="module")
def xxhash_64_decimal_tbl():
    arrow_tbl = pa.Table.from_arrays(
        [pa.array([0, 100, -100, 999999999, -999999999], type=pa.decimal(-3))],
        names=["data"],
    )
    return plc.interop.from_arrow(arrow_tbl)


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
            "struct": [{"a": 1}, {"a": 2}, {"a": 3}],
        }
    )
    return data


def all_types_output_table(input, method):
    def _applyfunc(x):
        hasher = getattr(hashlib, method)
        # TODO: not how libcudf computes row hash
        return hasher(str(x).encode()).hexdigest()

    result = pa.Table.from_pandas(input.to_pandas().map(_applyfunc))
    return result


def python_hash_value(x, method):
    if isinstance(x, str):
        binary = str(x).encode()
    elif isinstance(x, float):
        binary = struct.pack("<d", x)
    elif isinstance(x, bool):
        binary = x.to_bytes(1, byteorder="little", signed=True)
    elif isinstance(x, int):
        binary = x.to_bytes(8, byteorder="little", signed=True)
    elif isinstance(x, np.ndarray):
        binary = x.tobytes()
    else:
        raise NotImplementedError
    return getattr(hashlib, method)(binary).hexdigest()


@pytest.mark.parametrize(
    "method", ["sha1", "sha224", "sha256", "sha384", "sha512"]
)
def test_hash_column_sha(pa_input_column, method):
    plc_tbl = plc.interop.from_arrow(
        pa.Table.from_arrays([pa_input_column], names=["data"])
    )
    plc_hasher = getattr(plc.hashing, method)

    if isinstance(pa_input_column.type, (pa.ListType, pa.StructType)):
        with pytest.raises(TypeError):
            plc_hasher(plc_tbl)
        return

    expect = pa.Array.from_pandas(
        pa_input_column.to_pandas().apply(python_hash_value, args=(method,))
    )
    got = plc_hasher(plc_tbl)
    assert_column_eq(got, expect)


def test_hash_column_md5(pa_input_column):
    plc_tbl = plc.interop.from_arrow(
        pa.Table.from_arrays([pa_input_column], names=["data"])
    )

    if isinstance(pa_input_column.type, pa.StructType):
        with pytest.raises(TypeError):
            plc.hashing.md5(plc_tbl)
        return

    expect = pa.Array.from_pandas(
        pa_input_column.to_pandas().apply(python_hash_value, args=("md5",))
    )
    got = plc.hashing.md5(plc_tbl)
    assert_column_eq(got, expect)


@pytest.mark.parametrize(
    "method", ["sha1", "sha224", "sha256", "sha384", "sha512"]
)
@pytest.mark.parametrize("dtype", ["list", "struct"])
def test_sha_list_struct_err(all_types_input_table, dtype, method):
    err_types = all_types_input_table.select([dtype])
    plc_tbl = plc.interop.from_arrow(err_types)
    plc_hasher = getattr(plc.hashing, method)

    with pytest.raises(TypeError):
        plc_hasher(plc_tbl)


def test_xxhash_64_int(xxhash_64_int_tbl):
    expected = pa.array(
        [
            4827426872506142937,
            13867166853951622683,
            4246796580750024372,
            17339819992360460003,
            7292178400482025765,
            2971168436322821236,
            9380524276503839603,
            9380524276503839603,
        ],
        type=pa.uint64(),
    )
    got = plc.hashing.xxhash_64(xxhash_64_int_tbl, 0)
    assert_column_eq(got, expected)


def test_xxhash_64_double(xxhash_64_double_tbl):
    # see xxhash_64_test.cpp for details
    expected = pa.array(
        [
            16892115221677838993,
            1686446903308179321,
            3803688792395291579,
            18250447068822614389,
            3511911086082166358,
            4558309869707674848,
            18031741628920313605,
            16838308782748609196,
            3127544388062992779,
            1692401401506680154,
            13770442912356326755,
        ],
        type=pa.uint64(),
    )
    got = plc.hashing.xxhash_64(xxhash_64_double_tbl, 0)
    assert_column_eq(got, expected)


def test_xxhash_64_string(xxhash_64_string_tbl):
    def hasher(x):
        return xxhash.xxh64(bytes(x, "utf-8")).intdigest()

    expected = pa.Array.from_pandas(
        plc.interop.to_arrow(xxhash_64_string_tbl)
        .to_pandas()[""]
        .apply(hasher)
    )
    got = plc.hashing.xxhash_64(xxhash_64_string_tbl, 0)

    assert_column_eq(got, expected)


def test_xxhash64_decimal(xxhash_64_decimal_tbl):
    expected = pa.array(
        [
            4246796580750024372,
            5959467639951725378,
            4122185689695768261,
            3249245648192442585,
            8009575895491381648,
        ],
        type=pa.uint64(),
    )
    got = plc.hashing.xxhash_64(xxhash_64_decimal_tbl, 0)
    assert_table_eq(got, expected)
