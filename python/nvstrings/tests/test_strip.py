# Copyright (c) 2018-2019, NVIDIA CORPORATION.

import pandas as pd

import nvstrings
from utils import assert_eq


def test_strip():
    s = ["  hello  ", "  there  ", "  world  ", None, "  accénté  ", ""]
    strs = nvstrings.to_device(s)
    pstrs = pd.Series(s)
    got = strs.strip()
    expected = pstrs.str.strip()
    assert_eq(got.to_host(), expected)

    got = strs.strip().strip("é")
    expected = pstrs.str.strip().str.strip("é")
    assert_eq(got.to_host(), expected)

    got = strs.strip(" e")
    expected = pstrs.str.strip(" e")
    assert_eq(got.to_host(), expected)


def test_lstrip():
    s = ["  hello  ", "  there  ", "  world  ", None, "  accénté  ", ""]
    strs = nvstrings.to_device(s)
    pstrs = pd.Series(s)
    got = strs.lstrip()
    expected = pstrs.str.lstrip()
    assert_eq(got.to_host(), expected)


def test_rstrip():
    s = ["  hello  ", "  there  ", "  world  ", None, "  accénté  ", ""]
    strs = nvstrings.to_device(s)
    pstrs = pd.Series(s)
    got = strs.rstrip()
    expected = pstrs.str.rstrip()
    assert_eq(got.to_host(), expected)
