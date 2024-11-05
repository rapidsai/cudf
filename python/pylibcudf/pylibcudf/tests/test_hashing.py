# Copyright (c) 2024, NVIDIA CORPORATION.

import hashlib
import struct

import mmh3
import numpy as np
import pyarrow as pa
import pytest
import xxhash
from utils import assert_column_eq, assert_table_eq

import pylibcudf as plc

SEED = 0
METHODS = ["md5", "sha1", "sha224", "sha256", "sha384", "sha512"]


def scalar_to_binary(x):
    if isinstance(x, str):
        return x.encode()
    elif isinstance(x, float):
        return struct.pack("<d", x)
    elif isinstance(x, bool):
        return x.to_bytes(1, byteorder="little", signed=True)
    elif isinstance(x, int):
        return x.to_bytes(8, byteorder="little", signed=True)
    else:
        raise NotImplementedError


def hash_single_uint32(val, seed=0):
    return mmh3.hash(np.uint32(val).tobytes(), seed=seed, signed=False)


def hash_combine_32(lhs, rhs):
    return np.uint32(lhs ^ (rhs + 0x9E3779B9 + (lhs << 6) + (lhs >> 2)))


def uint_hash_combine_32(lhs, rhs):
    return hash_combine_32(np.uint32(lhs), np.uint32(rhs))


def libcudf_mmh3_x86_32(binary):
    seed = plc.hashing.LIBCUDF_DEFAULT_HASH_SEED
    hashval = mmh3.hash(binary, seed)
    return hash_combine_32(seed, hashval)


@pytest.fixture(params=[pa.int64(), pa.float64(), pa.string(), pa.bool_()])
def scalar_type(request):
    return request.param


@pytest.fixture
def pa_scalar_input_column(scalar_type):
    if pa.types.is_integer(scalar_type) or pa.types.is_floating(scalar_type):
        return pa.array([1, 2, 3], type=scalar_type)
    elif pa.types.is_string(scalar_type):
        return pa.array(["a", "b", "c"], type=scalar_type)
    elif pa.types.is_boolean(scalar_type):
        return pa.array([True, True, False], type=scalar_type)


@pytest.fixture
def plc_scalar_input_tbl(pa_scalar_input_column):
    return plc.interop.from_arrow(
        pa.Table.from_arrays([pa_scalar_input_column], names=["data"])
    )


@pytest.fixture(scope="module")
def list_struct_table():
    data = pa.Table.from_pydict(
        {
            "list": [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
            "struct": [{"a": 1, "b": 2}, {"a": 3, "b": 4}, {"a": 5, "b": 6}],
        }
    )
    return data


def python_hash_value(x, method):
    if method == "murmurhash3_x86_32":
        return libcudf_mmh3_x86_32(x)
    elif method == "murmurhash3_x64_128":
        hasher = mmh3.mmh3_x64_128(seed=plc.hashing.LIBCUDF_DEFAULT_HASH_SEED)
        hasher.update(x)
        # libcudf returns a tuple of two 64-bit integers
        return hasher.utupledigest()
    elif method == "xxhash_64":
        return xxhash.xxh64(
            x, seed=plc.hashing.LIBCUDF_DEFAULT_HASH_SEED
        ).intdigest()
    else:
        return getattr(hashlib, method)(x).hexdigest()


@pytest.mark.parametrize(
    "method", ["sha1", "sha224", "sha256", "sha384", "sha512", "md5"]
)
def test_hash_column_sha_md5(
    pa_scalar_input_column, plc_scalar_input_tbl, method
):
    plc_hasher = getattr(plc.hashing, method)

    def py_hasher(val):
        return getattr(hashlib, method)(scalar_to_binary(val)).hexdigest()

    expect = pa.array(
        [py_hasher(val) for val in pa_scalar_input_column.to_pylist()],
        type=pa.string(),
    )
    got = plc_hasher(plc_scalar_input_tbl)
    assert_column_eq(got, expect)


def test_hash_column_xxhash64(pa_scalar_input_column, plc_scalar_input_tbl):
    def py_hasher(val):
        return xxhash.xxh64(
            scalar_to_binary(val), seed=plc.hashing.LIBCUDF_DEFAULT_HASH_SEED
        ).intdigest()

    expect = pa.array(
        [py_hasher(val) for val in pa_scalar_input_column.to_pylist()],
        type=pa.uint64(),
    )
    got = plc.hashing.xxhash_64(plc_scalar_input_tbl, 0)

    assert_column_eq(got, expect)


@pytest.mark.parametrize(
    "method", ["sha1", "sha224", "sha256", "sha384", "sha512"]
)
@pytest.mark.parametrize("dtype", ["list", "struct"])
def test_sha_list_struct_err(list_struct_table, dtype, method):
    err_types = list_struct_table.select([dtype])
    plc_tbl = plc.interop.from_arrow(err_types)
    plc_hasher = getattr(plc.hashing, method)

    with pytest.raises(TypeError):
        plc_hasher(plc_tbl)


def test_md5_struct_err(list_struct_table):
    err_types = list_struct_table.select(["struct"])
    plc_tbl = plc.interop.from_arrow(err_types)

    with pytest.raises(TypeError):
        plc.hashing.md5(plc_tbl)


def test_murmurhash3_x86_32(pa_scalar_input_column, plc_scalar_input_tbl):
    def py_hasher(val):
        return libcudf_mmh3_x86_32(scalar_to_binary(val))

    got = plc.hashing.murmurhash3_x86_32(plc_scalar_input_tbl, 0)
    expect = pa.array(
        [py_hasher(val) for val in pa_scalar_input_column.to_pylist()],
        type=pa.uint32(),
    )
    got = plc.hashing.murmurhash3_x86_32(plc_scalar_input_tbl, 0)
    assert_column_eq(got, expect)


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_murmurhash3_x86_32_list():
    pa_tbl = pa.Table.from_pydict(
        {
            "list": pa.array(
                [[1, 2, 3], [4, 5, 6], [7, 8, 9]], type=pa.list_(pa.uint32())
            )
        }
    )
    plc_tbl = plc.interop.from_arrow(pa_tbl)

    def hash_list(list_):
        hash_value = uint_hash_combine_32(0, hash_single_uint32(len(list_)))

        for element in list_:
            hash_value = uint_hash_combine_32(
                hash_value,
                hash_single_uint32(
                    element, seed=plc.hashing.LIBCUDF_DEFAULT_HASH_SEED
                ),
            )

        final = uint_hash_combine_32(
            plc.hashing.LIBCUDF_DEFAULT_HASH_SEED, hash_value
        )
        return final

    expect = pa.array(
        [hash_list(val) for val in pa_tbl["list"].to_pylist()],
        type=pa.uint32(),
    )
    got = plc.hashing.murmurhash3_x86_32(
        plc_tbl, plc.hashing.LIBCUDF_DEFAULT_HASH_SEED
    )
    assert_column_eq(got, expect)


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_murmurhash3_x86_32_struct():
    pa_tbl = pa.table(
        {
            "struct": pa.array(
                [
                    {"a": 1, "b": 2, "c": 3},
                    {"a": 4, "b": 5, "c": 6},
                    {"a": 7, "b": 8, "c": 9},
                ],
                type=pa.struct(
                    [
                        pa.field("a", pa.uint32()),
                        pa.field("b", pa.uint32(), pa.field("c", pa.uint32())),
                    ]
                ),
            )
        }
    )
    plc_tbl = plc.interop.from_arrow(pa_tbl)

    def hash_struct(s):
        seed = plc.hashing.LIBCUDF_DEFAULT_HASH_SEED
        keys = list(s.keys())

        combined_hash = hash_single_uint32(s[keys[0]], seed=seed)
        combined_hash = uint_hash_combine_32(0, combined_hash)
        combined_hash = uint_hash_combine_32(seed, combined_hash)

        for key in keys[1:]:
            current_hash = hash_single_uint32(s[key], seed=seed)
            combined_hash = uint_hash_combine_32(combined_hash, current_hash)

        return combined_hash

    got = plc.hashing.murmurhash3_x86_32(
        plc_tbl, plc.hashing.LIBCUDF_DEFAULT_HASH_SEED
    )

    expect = pa.array(
        [hash_struct(val) for val in pa_tbl["struct"].to_pylist()],
        type=pa.uint32(),
    )
    assert_column_eq(got, expect)


def test_murmurhash3_x64_128(pa_scalar_input_column, plc_scalar_input_tbl):
    def py_hasher(val):
        hasher = mmh3.mmh3_x64_128(seed=plc.hashing.LIBCUDF_DEFAULT_HASH_SEED)
        hasher.update(val)
        return hasher.utupledigest()

    tuples = [
        py_hasher(scalar_to_binary(val))
        for val in pa_scalar_input_column.to_pylist()
    ]
    expect = pa.Table.from_arrays(
        [
            pa.array([np.uint64(t[0]) for t in tuples]),
            pa.array([np.uint64(t[1]) for t in tuples]),
        ],
        names=["0", "1"],
    )
    got = plc.hashing.murmurhash3_x64_128(plc_scalar_input_tbl, 0)

    assert_table_eq(expect, got)
