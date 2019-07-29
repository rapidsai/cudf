# Copyright (c) 2018-2019, NVIDIA CORPORATION.

import nvstrings
from utils import assert_eq


def test_lower():
    strs = nvstrings.to_device(["abc", "Def", None, "jLl"])
    got = strs.lower()
    expected = ['abc', 'def', None, 'jll']
    assert_eq(got, expected)


def test_upper():
    strs = nvstrings.to_device(["abc", "Def", None, "jLl"])
    got = strs.upper()
    expected = ['ABC', 'DEF', None, 'JLL']
    assert_eq(got, expected)


def test_swapcase():
    strs = nvstrings.to_device(["abc", "Def", None, "jLl"])
    got = strs.swapcase()
    expected = ['ABC', 'dEF', None, 'JlL']
    assert_eq(got, expected)


def test_capitalize():
    strs = nvstrings.to_device(["abc", "Def", None, "jLl"])
    got = strs.capitalize()
    expected = ['Abc', 'Def', None, 'Jll']
    assert_eq(got, expected)


def test_title():
    strs = nvstrings.to_device(["abc", "Def", None, "jLl"])
    got = strs.title()
    expected = ['Abc', 'Def', None, 'Jll']
    assert_eq(got, expected)


def test_islower():
    strs = nvstrings.to_device(["abc", "Def", None, "jLl"])
    got = strs.islower()
    expected = [True, False, None, False]
    assert_eq(got, expected)


def test_isupper():
    strs = nvstrings.to_device(["abc", "Def", None, "jLl"])
    got = strs.isupper()
    expected = [False, False, None, False]
    assert_eq(got, expected)
