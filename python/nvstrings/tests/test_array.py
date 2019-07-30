# Copyright (c) 2018-2019, NVIDIA CORPORATION.

import nvstrings


def test_gather():
    strs = nvstrings.to_device(["abc", "defghi", None, "cat"])
    got = strs.gather([1, 3, 2])
    expected = ["defghi", "cat", None]
    assert got.to_host() == expected


def test_gather_bool():
    strs = nvstrings.to_device(["abc", "defghi", None, "cat"])
    got = strs.gather([True, False, False, True])
    expected = ["abc", "cat"]
    assert got.to_host() == expected


def test_sublist():
    strs = nvstrings.to_device(["abc", "defghi", None, "cat"])
    got = strs.sublist([1, 3, 2])
    expected = ["defghi", "cat", None]
    assert got.to_host() == expected


def test_remove_strings():
    strs = nvstrings.to_device(["abc", "defghi", None, "cat"])
    got = strs.remove_strings([0, 2])
    expected = ["defghi", "cat"]
    assert got.to_host() == expected


def test_scatter():
    s1 = nvstrings.to_device(["a", "b", "c", "d"])
    s2 = nvstrings.to_device(["e", "f"])
    got = s1.scatter(s2, [1, 3])
    expected = ["a", "e", "c", "f"]
    assert got.to_host() == expected


def test_scalar_scatter():
    s1 = nvstrings.to_device(["a", "b", "c", "d"])
    got = s1.scalar_scatter("+", [1, 3], 2)
    expected = ["a", "+", "c", "+"]
    assert got.to_host() == expected
