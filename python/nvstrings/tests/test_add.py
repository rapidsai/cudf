# Copyright (c) 2018-2019, NVIDIA CORPORATION.

import nvstrings


def test_from_strings():
    s1 = nvstrings.to_device(["dog and cat", None, "accénted", ""])
    got = nvstrings.from_strings(s1, s1)
    expected = ['dog and cat', None, 'accénted', '',
                'dog and cat', None, 'accénted', '']
    assert got.to_host() == expected


def test_add_strings():
    s1 = nvstrings.to_device(["dog and cat", None, "accénted", ""])
    s2 = nvstrings.to_device(["aaa", None, "", "bbb"])
    got = s1.add_strings(s2)
    expected = ['dog and cat', None, 'accénted', '', 'aaa', None, '', 'bbb']
    assert got.to_host() == expected
