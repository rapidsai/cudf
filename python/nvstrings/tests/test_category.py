# Copyright (c) 2018-2019, NVIDIA CORPORATION.

import nvstrings
import nvcategory

import numpy as np
import pytest

from utils import assert_eq


def test_size():
    strs = nvstrings.to_device(
        ["eee", "aaa", "eee", "ddd", "ccc", "ccc", "ccc", "eee", "aaa"]
    )
    cat = nvcategory.from_strings(strs)
    assert strs.size() == cat.size()


def test_keys():
    strs1 = nvstrings.to_device(["a", "b", "b", "f", "c", "f"])
    cat = nvcategory.from_strings(strs1)
    got = cat.keys()
    expected = ["a", "b", "c", "f"]
    assert_eq(got, expected)


def test_keys_size():
    strs1 = nvstrings.to_device(["a", "b", "b", "f", "c", "f"])
    cat = nvcategory.from_strings(strs1)
    got = cat.keys_size()
    assert got == 4


def test_values():
    strs = nvstrings.to_device(
        ["eee", "aaa", "eee", "ddd", "ccc", "ccc", "ccc", "eee", "aaa"]
    )
    cat = nvcategory.from_strings(strs)
    got = cat.values()
    expected = [3, 0, 3, 2, 1, 1, 1, 3, 0]
    assert_eq(got, expected)


def test_value_for_index():
    strs = nvstrings.to_device(
        ["eee", "aaa", "eee", "ddd", "ccc", "ccc", "ccc", "eee", "aaa"]
    )
    cat = nvcategory.from_strings(strs)
    got = cat.value_for_index(7)
    expected = 3
    assert got == expected


def test_value():
    strs = nvstrings.to_device(
        ["eee", "aaa", "eee", "ddd", "ccc", "ccc", "ccc", "eee", "aaa"]
    )
    cat = nvcategory.from_strings(strs)
    got = cat.value("ccc")
    expected = 1
    assert got == expected


def test_indexes_for_key():
    strs = nvstrings.to_device(
        ["eee", "aaa", "eee", "ddd", "ccc", "ccc", "ccc", "eee", "aaa"]
    )
    cat = nvcategory.from_strings(strs)
    got = cat.indexes_for_key("ccc")
    expected = [4, 5, 6]
    assert_eq(got, expected)


def test_to_strings():
    strs = nvstrings.to_device(
        ["eee", "aaa", "eee", "ddd", "ccc", "ccc", "ccc", "eee", "aaa"]
    )
    cat = nvcategory.from_strings(strs)
    got = cat.to_strings()
    assert_eq(got, strs)


def test_add_strings():
    strs = nvstrings.to_device(
        ["eee", "aaa", "eee", "ddd", "ccc", "ccc", "ccc", "eee", "aaa"]
    )
    cat = nvcategory.from_strings(strs)
    got = cat.add_strings(strs)
    expected_keys = ["aaa", "ccc", "ddd", "eee"]
    expected_values = [3, 0, 3, 2, 1, 1, 1, 3, 0, 3, 0, 3, 2, 1, 1, 1, 3, 0]
    assert_eq(got.keys(), expected_keys)
    assert_eq(got.values(), expected_values)


def test_gather_strings():
    strs = nvstrings.to_device(
        ["eee", "aaa", "eee", "ddd", "ccc", "ccc", "ccc", "eee", "aaa"]
    )
    cat = nvcategory.from_strings(strs)
    got = cat.gather_strings([0, 2, 0])
    expected = ["aaa", "ddd", "aaa"]
    assert_eq(got, expected)


@pytest.mark.parametrize(
    "func",
    [
        lambda cat, indexes: cat.gather_strings(indexes),
        lambda cat, indexes: cat.gather(indexes),
        lambda cat, indexes: cat.gather_and_remap(indexes),
    ],
)
def test_gather_index_exception(func):
    strs = nvstrings.to_device(
        ["eee", "aaa", "eee", "ddd", "ccc", "ccc", "ccc", "eee", "aaa"]
    )
    cat = nvcategory.from_strings(strs)
    indexes = [0, 2, 0, 4]
    with pytest.raises(Exception):
        func(cat, indexes)


def test_remove_strings():
    strs = nvstrings.to_device(
        ["eee", "aaa", "eee", "ddd", "ccc", "ccc", "ccc", "eee", "aaa"]
    )
    cat = nvcategory.from_strings(strs)
    removal_strings = nvstrings.to_device(["ccc", "aaa", "bbb"])
    got = cat.remove_strings(removal_strings)

    expected_keys = ["ddd", "eee"]
    expected_values = [1, 1, 0, 1]
    assert_eq(got.keys(), expected_keys)
    assert_eq(got.values(), expected_values)


def test_from_strings():
    strs1 = nvstrings.to_device(
        ["eee", "aaa", "eee", "ddd", "ccc", "ccc", "ccc", "eee", "aaa"]
    )
    strs2 = nvstrings.to_device(
        ["ggg", "fff", "hhh", "aaa", "fff", "fff", "ggg", "hhh", "bbb"]
    )
    cat = nvcategory.from_strings(strs1, strs2)

    expected_keys = ["aaa", "bbb", "ccc", "ddd", "eee", "fff", "ggg", "hhh"]
    expected_values = [4, 0, 4, 3, 2, 2, 2, 4, 0, 6, 5, 7, 0, 5, 5, 6, 7, 1]
    assert_eq(cat.keys(), expected_keys)
    assert_eq(cat.values(), expected_values)


def test_merge_category():
    strs1 = nvstrings.to_device(
        ["eee", "aaa", "eee", "ddd", "ccc", "ccc", "ccc", "eee", "aaa"]
    )
    strs2 = nvstrings.to_device(
        ["ggg", "fff", "hhh", "aaa", "fff", "fff", "ggg", "hhh", "bbb"]
    )
    cat1 = nvcategory.from_strings(strs1)
    cat2 = nvcategory.from_strings(strs2)
    ncat = cat1.merge_category(cat2)

    expected_keys = ["aaa", "ccc", "ddd", "eee", "bbb", "fff", "ggg", "hhh"]
    expected_values = [3, 0, 3, 2, 1, 1, 1, 3, 0, 6, 5, 7, 0, 5, 5, 6, 7, 4]
    assert_eq(ncat.keys(), expected_keys)
    assert_eq(ncat.values(), expected_values)


def test_merge_and_remap():
    strs1 = nvstrings.to_device(
        ["eee", "aaa", "eee", "ddd", "ccc", "ccc", "ccc", "eee", "aaa"]
    )
    strs2 = nvstrings.to_device(
        ["ggg", "fff", "hhh", "aaa", "fff", "fff", "ggg", "hhh", "bbb"]
    )
    cat1 = nvcategory.from_strings(strs1)
    cat2 = nvcategory.from_strings(strs2)
    ncat = cat1.merge_and_remap(cat2)

    expected_keys = ["aaa", "bbb", "ccc", "ddd", "eee", "fff", "ggg", "hhh"]
    expected_values = [4, 0, 4, 3, 2, 2, 2, 4, 0, 6, 5, 7, 0, 5, 5, 6, 7, 1]
    assert_eq(ncat.keys(), expected_keys)
    assert_eq(ncat.values(), expected_values)


def test_add_keys():
    strs1 = nvstrings.to_device(["a", "b", "b", "f", "c", "f"])
    strs2 = nvstrings.to_device(["a", "b", "c", "d"])
    cat = nvcategory.from_strings(strs1)
    cat1 = cat.add_keys(strs2)
    assert_eq(cat1.keys(), ["a", "b", "c", "d", "f"])


def test_remove_keys():
    strs1 = nvstrings.to_device(["a", "b", "b", "f", "c", "f"])
    strs2 = nvstrings.to_device(["b", "d"])
    cat = nvcategory.from_strings(strs1)
    cat1 = cat.remove_keys(strs2)
    assert_eq(cat1.keys(), ["a", "c", "f"])


def test_set_keys():
    strs1 = nvstrings.to_device(["a", "b", "b", "f", "c", "f"])
    strs2 = nvstrings.to_device(["b", "c", "e", "d"])
    cat = nvcategory.from_strings(strs1)
    cat1 = cat.set_keys(strs2)
    assert_eq(cat1.keys(), ["b", "c", "d", "e"])


def test_remove_unused_keys():
    strs1 = nvstrings.to_device(["a", "b", "b", "f", "c", "f"])
    strs2 = nvstrings.to_device(["b", "c", "e", "d"])
    cat = nvcategory.from_strings(strs1)
    cat1 = cat.set_keys(strs2)
    cat1_unused_removed = cat1.remove_unused_keys()
    assert_eq(cat1_unused_removed.keys(), ["b", "c"])


def test_gather():
    strs1 = nvstrings.to_device(["a", "b", "b", "f", "c", "f"])
    cat = nvcategory.from_strings(strs1)
    cat1 = cat.gather([1, 3, 2, 3, 1, 2])

    expected_keys = ["a", "b", "c", "f"]
    expected_values = [1, 3, 2, 3, 1, 2]
    assert_eq(cat1.keys(), expected_keys)
    assert_eq(cat1.values(), expected_values)


def test_gather_and_remap():
    strs1 = nvstrings.to_device(["a", "b", "b", "f", "c", "f"])
    cat = nvcategory.from_strings(strs1)
    cat1 = cat.gather_and_remap([1, 3, 2, 3, 1, 2])

    expected_keys = ["b", "c", "f"]
    expected_values = [0, 2, 1, 2, 0, 1]
    assert_eq(cat1.keys(), expected_keys)
    assert_eq(cat1.values(), expected_values)


def test_from_offsets():
    values = np.array([97, 112, 112, 108, 101], dtype=np.int8)
    offsets = np.array([0, 1, 2, 3, 4, 5], dtype=np.int32)
    cat = nvcategory.from_offsets(values, offsets, 5)
    expected_keys = ["a", "e", "l", "p"]
    expected_values = [0, 3, 3, 2, 1]
    assert_eq(cat.keys(), expected_keys)
    assert_eq(cat.values(), expected_values)


def test_from_strings_list():
    s1 = nvstrings.to_device(["apple", "pear", "banana"])
    s2 = nvstrings.to_device(["orange", "pear"])
    cat = nvcategory.from_strings_list([s1, s2])

    expected_keys = ["apple", "banana", "orange", "pear"]
    expected_values = [0, 3, 1, 2, 3]
    assert_eq(cat.keys(), expected_keys)
    assert_eq(cat.values(), expected_values)


def test_to_device():
    cat = nvcategory.to_device(["apple", "pear", "banana", "orange", "pear"])
    expected_keys = ["apple", "banana", "orange", "pear"]
    expected_values = [0, 3, 1, 2, 3]
    assert_eq(cat.keys(), expected_keys)
    assert_eq(cat.values(), expected_values)
