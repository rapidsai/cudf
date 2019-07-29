# Copyright (c) 2019, NVIDIA CORPORATION.

import nvcategory
import numpy as np
from utils import assert_eq


def test_size():
    narr = np.array([4, 1, 2, 3, 2, 1, 4, 1, 1])
    cat = nvcategory.from_numbers(narr)
    assert narr.size == cat.size()


def test_keys_size():
    narr = np.array([4, 1, 2, 3, 2, 1, 4, 1, 1], dtype=np.int32)
    cat = nvcategory.from_numbers(narr)
    got = cat.keys_size()
    assert got == 4


def test_keys():
    narr = np.array([2, 1, 1.25, 1.5, 1, 1.25, 1, 1, 2])
    cat = nvcategory.from_numbers(narr)
    keys = np.empty([cat.keys_size()], dtype=narr.dtype)
    cat.keys(keys)
    got = keys.tolist()
    expected = [1.0, 1.25, 1.5, 2.0]
    assert_eq(got, expected)


def test_values():
    narr = np.array([4, 1, 2, 3, 2, 1, 4, 1, 1])
    cat = nvcategory.from_numbers(narr)
    values = np.empty([cat.size()], dtype=np.int32)
    cat.values(values)
    got = values.tolist()
    expected = [3, 0, 1, 2, 1, 0, 3, 0, 0]
    assert_eq(got, expected)


def test_indexes_for_key():
    narr = np.array([4, 1, 2, 3, 2, 1, 4, 1, 1])
    cat = nvcategory.from_numbers(narr)
    count = cat.indexes_for_key(1)
    assert count == 4
    idxs = np.empty([count], dtype=np.int32)
    cat.indexes_for_key(1, idxs)
    got = idxs.tolist()
    expected = [1, 5, 7, 8]
    assert_eq(got, expected)


def test_to_numbers():
    narr = np.array([2, 1, 1.25, 1.5, 1, 1.25, 1, 1, 2])
    cat = nvcategory.from_numbers(narr)
    nbrs = np.empty([cat.size()], dtype=narr.dtype)
    cat.to_numbers(nbrs)
    got = nbrs.tolist()
    expected = narr.tolist()
    assert_eq(got, expected)


def test_gather_numbers():
    narr = np.array([1, 1.25, 1.5, 1, 1.25, 1, 1, 2])
    cat = nvcategory.from_numbers(narr)
    idxs = np.array([0, 2, 0], dtype=np.int32)
    nbrs = np.empty([idxs.size], dtype=narr.dtype)
    cat.gather_numbers(idxs, nbrs)
    got = nbrs.tolist()
    expected = [1., 1.5, 1.]
    assert_eq(got, expected)


def util_check_cat(ncat, dtype):
    keys = np.empty([ncat.keys_size()], dtype=dtype)
    values = np.empty([ncat.size()], dtype=np.int32)
    ncat.keys(keys)
    ncat.values(values)
    got_keys = keys.tolist()
    got_values = values.tolist()
    return (got_keys, got_values)


def test_merge_category():
    cat1 = nvcategory.from_numbers(
        np.array([4, 1, 2, 3, 2, 1, 4, 1, 1]))
    cat2 = nvcategory.from_numbers(np.array([2, 4, 3, 0]))
    ncat = cat1.merge_and_remap(cat2)

    (got_keys, got_values) = util_check_cat(ncat, np.int64)
    expected_keys = [0, 1, 2, 3, 4]
    expected_values = [4, 1, 2, 3, 2, 1, 4, 1, 1, 2, 4, 3, 0]
    assert_eq(got_keys, expected_keys)
    assert_eq(got_values, expected_values)


def test_add_keys():
    narr = np.array([2, 1, 1.25, 1.5, 1, 1.25, 1, 1, 2])
    cat = nvcategory.from_numbers(narr)
    ncat = cat.add_keys(np.array([2, 1, 1.75, 0]))

    (got_keys, got_values) = util_check_cat(ncat, narr.dtype)
    expected_keys = [0, 1, 1.25, 1.5, 1.75, 2]
    expected_values = [5, 1, 2, 3, 1, 2, 1, 1, 5]
    assert_eq(got_keys, expected_keys)
    assert_eq(got_values, expected_values)


def test_remove_keys():
    narr = np.array([4, 1, 2, 3, 2, 1, 4, 1, 1])
    cat = nvcategory.from_numbers(narr)
    ncat = cat.remove_keys(np.array([3, 0]))

    (got_keys, got_values) = util_check_cat(ncat, narr.dtype)
    expected_keys = [1, 2, 4]
    expected_values = [2, 0, 1, -1, 1, 0, 2, 0, 0]
    assert_eq(got_keys, expected_keys)
    assert_eq(got_values, expected_values)


def test_set_keys():
    narr = np.array([4, 1, 2, 3, 2, 1, 4, 1, 1], dtype=np.int8)
    cat = nvcategory.from_numbers(narr)
    ncat = cat.set_keys(np.array([2, 4, 3, 0], dtype=narr.dtype))

    (got_keys, got_values) = util_check_cat(ncat, narr.dtype)
    expected_keys = [0, 2, 3, 4]
    expected_values = [3, -1, 1, 2, 1, -1, 3, -1, -1]
    assert_eq(got_keys, expected_keys)
    assert_eq(got_values, expected_values)


def test_remove_unused_keys():
    narr = np.array([4, 1, 2, 3, 2, 1, 4, 1, 1], dtype=np.int32)
    cat = nvcategory.from_numbers(narr)
    ncat = cat.add_keys(np.array([2, 4, 3, 0], dtype=narr.dtype))
    ncat = ncat.remove_unused_keys()

    (got_keys, got_values) = util_check_cat(ncat, narr.dtype)
    expected_keys = [1, 2, 3, 4]
    expected_values = [3, 0, 1, 2, 1, 0, 3, 0, 0]
    assert_eq(got_keys, expected_keys)
    assert_eq(got_values, expected_values)


def test_gather():
    narr = np.array([2, 1, 1.25, 1.5, 1, 1.25, 1, 1, 2], dtype=np.float32)
    cat = nvcategory.from_numbers(narr)
    ncat = cat.gather(np.array([1, 3, 2, 3, 1, 2], dtype=np.int32))

    (got_keys, got_values) = util_check_cat(ncat, narr.dtype)
    expected_keys = [1.0, 1.25, 1.5, 2.0]
    expected_values = [1, 3, 2, 3, 1, 2]
    assert_eq(got_keys, expected_keys)
    assert_eq(got_values, expected_values)


def test_gather_and_remap():
    narr = np.array([4, 1, 2, 3, 2, 1, 4, 1, 1], dtype=np.float32)
    cat = nvcategory.from_numbers(narr)
    ncat = cat.gather_and_remap(np.array([1, 3, 2, 3, 1, 2], dtype=np.int32))

    (got_keys, got_values) = util_check_cat(ncat, narr.dtype)
    expected_keys = [2.0, 3.0, 4.0]
    expected_values = [0, 2, 1, 2, 0, 1]
    assert_eq(got_keys, expected_keys)
    assert_eq(got_values, expected_values)


def test_nulls():
    narr = np.array([4, 1, 2, 3, 2, 1, 4, 1, 1])
    bitmask = np.array([1+2+8+32+64+128, 1], dtype=np.int8)
    cat = nvcategory.from_numbers(narr, bitmask)

    (got_keys, got_values) = util_check_cat(cat, narr.dtype)
    expected_keys = [2, 1, 3, 4]
    expected_values = [3, 1, 0, 2, 0, 1, 3, 1, 1]
    assert_eq(got_keys, expected_keys)
    assert_eq(got_values, expected_values)
