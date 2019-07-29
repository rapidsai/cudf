# Copyright (c) 2018-2019, NVIDIA CORPORATION.

import numpy as np
import pandas as pd
import nvstrings

from utils import assert_eq


def compare_split_records(nvstrs, pstrs):
    for i in range(len(nvstrs)):
        got = nvstrs[i]
        expected = pstrs[i]
        if not got:
            if expected is None:
                continue
        assert got.to_host() == expected


def test_split_record():
    s = ["héllo", None, "a_bc_déf", "a__bc", "_ab_cd", "ab_cd_", "", ' a b ']
    strs = nvstrings.to_device(s)
    pstrs = pd.Series(s)
    nstrs = strs.split_record("_")
    ps = pstrs.str.split('_')
    compare_split_records(nstrs, ps)


def test_split():
    s = ["héllo", None, "a_bc_déf", "a__bc", "_ab_cd", "ab_cd_", "", ' a b ',
         ' a  bbb   c']
    strs = nvstrings.to_device(s)
    got = strs.split("_")
    expected = np.array(
        [['héllo', None, 'a', 'a', '', 'ab', '', ' a b ', ' a  bbb   c'],
         [None, None, 'bc', '', 'ab', 'cd', None, None, None],
         [None, None, 'déf', 'bc', 'cd', '', None, None, None]])

    for i in range(len(got)):
        assert_eq(got[i], expected[i])


def test_rsplit():
    s = ["héllo", None, "a_bc_déf", "a__bc", "_ab_cd", "ab_cd_", "", ' a b ',
         ' a  bbb   c']
    strs = nvstrings.to_device(s)
    got = strs.rsplit("_")
    expected = np.array([
        ['héllo', None, 'a', 'a', '', 'ab', '', ' a b ', ' a  bbb   c'],
        [None, None, 'bc', '', 'ab', 'cd', None, None, None],
        [None, None, 'déf', 'bc', 'cd', '', None, None, None],
    ])
    for i in range(len(got)):
        assert_eq(got[i], expected[i])


def test_rsplit_record():
    s = ["héllo", None, "a_bc_déf", "a__bc", "_ab_cd", "ab_cd_", "", ' a b ',
         ' a  bbb   c']
    strs = nvstrings.to_device(s)
    pstrs = pd.Series(s)
    nstrs = strs.rsplit_record("_")
    ps = pstrs.str.rsplit('_')
    compare_split_records(nstrs, ps)


def test_partition():
    s = ["héllo", None, "a_bc_déf", "a__bc", "_ab_cd", "ab_cd_", "", ' a b ',
         ' a  bbb   c']
    strs = nvstrings.to_device(s)
    got = strs.partition("_")
    expected = np.array([
        ['héllo', '', ''],
        [None, None, None],
        ['a', '_', 'bc_déf'],
        ['a', '_', '_bc'],
        ['', '_', 'ab_cd'],
        ['ab', '_', 'cd_'],
        ['', '', ''],
        [' a b ', '', ''],
        [' a  bbb   c', '', ''],
        ])
    for i in range(len(got)):
        assert_eq(got[i], expected[i])


def test_rpartition():
    s = ["héllo", None, "a_bc_déf", "a__bc", "_ab_cd", "ab_cd_", "", ' a b ',
         ' a  bbb   c']
    strs = nvstrings.to_device(s)
    got = strs.rpartition("_")
    expected = np.array([
        ['', '', 'héllo'],
        [None, None, None],
        ['a_bc', '_', 'déf'],
        ['a_', '_', 'bc'],
        ['_ab', '_', 'cd'],
        ['ab_cd', '_', ''],
        ['', '', ''],
        ['', '', ' a b '],
        ['', '', ' a  bbb   c'],
        ])
    for i in range(len(got)):
        assert_eq(got[i], expected[i])
