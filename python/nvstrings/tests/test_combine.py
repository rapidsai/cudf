# Copyright (c) 2018-2019, NVIDIA CORPORATION.

import nvstrings
from utils import assert_eq


def test_cat():
    strs = nvstrings.to_device(
        ["abc", "def", None, "", "jkl", "mno", "accént"]
    )
    got = strs.cat()
    expected = ["abcdefjklmnoaccént"]
    assert_eq(got, expected)

    # non-default separator
    got = strs.cat(sep=":")
    expected = ["abc:def::jkl:mno:accént"]
    assert_eq(got, expected)

    # non default separator and na_rep
    got = strs.cat(sep=":", na_rep="_")
    expected = ["abc:def:_::jkl:mno:accént"]
    assert_eq(got, expected)

    # non-null others, default separator, and na_rep
    strs2 = nvstrings.to_device(["1", "2", "3", "4", "5", "é", None])
    got = strs.cat(strs2, sep=":", na_rep="_")
    expected = ["abc:1", "def:2", "_:3", ":4", "jkl:5", "mno:é", "accént:_"]
    assert_eq(got, expected)

    # nvstrings others
    strs2 = nvstrings.to_device(["1", "2", "3", None, "5", "é", ""])
    got = strs.cat(strs2)
    expected = ["abc1", "def2", None, None, "jkl5", "mnoé", "accént"]
    assert_eq(got, expected)


def test_cat_multiple():
    strs = nvstrings.to_device(["abc", "df", None, "", "jkl", "mn", "accént"])
    strs1 = nvstrings.to_device(["1", "2", "3", "4", "5", "é", None])
    strs2 = nvstrings.to_device(["1", "2", "3", None, "5", "é", ""])
    got = strs.cat([strs1, strs2])
    expected = ["abc11", "df22", None, None, "jkl55", "mnéé", None]
    assert_eq(got, expected)

    got = strs.cat([strs1, strs2], sep=":", na_rep="_")
    expected = [
        "abc:1:1",
        "df:2:2",
        "_:3:3",
        ":4:_",
        "jkl:5:5",
        "mn:é:é",
        "accént:_:",
    ]
    assert_eq(got, expected)


def test_join():
    strs = nvstrings.to_device(["1", "2", "3", None, "5", "é", ""])
    got = strs.join()
    expected = ["1235é"]
    assert_eq(got, expected)

    # non-default sep
    got = strs.join(sep=":")
    expected = ["1:2:3:5:é:"]
    assert_eq(got, expected)
