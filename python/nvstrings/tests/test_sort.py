# Copyright (c) 2018-2019, NVIDIA CORPORATION.

import nvstrings

from utils import assert_eq


def test_sort_length():
    strs = nvstrings.to_device(
        ["abc", "defghi", None, "jkl", "mno", "pqr", "stu", "dog and cat",
         "accénted", ""])
    sorted_strs = strs.sort(1)
    expected = [None, '', 'abc', 'jkl', 'mno', 'pqr', 'stu', 'defghi',
                'accénted', 'dog and cat']
    assert_eq(sorted_strs, expected)


def test_sort_alphabetical():
    strs = nvstrings.to_device(
        ["abc", "defghi", None, "jkl", "mno", "pqr", "stu", "dog and cat",
         "accénted", ""])
    sorted_strs = strs.sort()
    expected = [None, '', 'abc', 'accénted', 'defghi', 'dog and cat', 'jkl',
                'mno', 'pqr', 'stu']
    assert_eq(sorted_strs, expected)


def test_sort_length_alphabetical():
    strs = nvstrings.to_device(
        ["abc", "defghi", None, "jkl", "mno", "pqr", "stu", "dog and cat",
         "accénted", ""])
    sorted_strs = strs.sort(3)
    expected = [None, '', 'abc', 'jkl', 'mno', 'pqr', 'stu', 'defghi',
                'accénted', 'dog and cat']
    assert_eq(sorted_strs, expected)


def test_order_length():
    strs = nvstrings.to_device(
        ["abc", "defghi", None, "jkl", "mno", "pqr", "stu", "dog and cat",
         "accénted", ""])
    sorted_strs = strs.order(1)
    expected = [2, 9, 0, 3, 4, 5, 6, 1, 8, 7]
    assert_eq(sorted_strs, expected)


def test_order_alphabetical():
    strs = nvstrings.to_device(
        ["abc", "defghi", None, "jkl", "mno", "pqr", "stu", "dog and cat",
         "accénted", ""])
    sorted_strs = strs.order()
    expected = [2, 9, 0, 8, 1, 7, 3, 4, 5, 6]
    assert_eq(sorted_strs, expected)


def test_order_length_alphabetical():
    strs = nvstrings.to_device(
        ["abc", "defghi", None, "jkl", "mno", "pqr", "stu", "dog and cat",
         "accénted", ""])
    sorted_strs = strs.order(3)
    expected = [2, 9, 0, 3, 4, 5, 6, 1, 8, 7]
    assert_eq(sorted_strs, expected)
