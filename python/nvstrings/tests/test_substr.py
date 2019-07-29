# Copyright (c) 2018-2019, NVIDIA CORPORATION.

import pytest
import numpy as np
import pandas as pd
import nvstrings

from librmm_cffi import librmm as rmm

from utils import assert_eq


def test_slice_from():
    strs = nvstrings.to_device(
        ["hello world", "holy accéntéd", "batman", None, ""])
    d_arr = rmm.to_device(np.asarray([2, 3, -1, -1, -1], dtype=np.int32))
    got = strs.slice_from(starts=d_arr.device_ctypes_pointer.value)
    expected = ['llo world', 'y accéntéd', '', None, '']
    assert_eq(got, expected)


@pytest.mark.parametrize('start', [2, 2, 2, 2])
@pytest.mark.parametrize('stop', [8, 15, 8, 8])
@pytest.mark.parametrize('step', [None, None, 2, 5])
def test_slice(start, stop, step):
    s = ["abcdefghij", "0123456789", "9876543210", None, "accénted", ""]
    strs = nvstrings.to_device(s)
    pstrs = pd.Series(s)
    got = strs.slice(start, stop, step)
    expected = pstrs.str.slice(start, stop, step)
    assert_eq(got.to_host(), expected)


@pytest.mark.parametrize('start', [2, 5])
@pytest.mark.parametrize('stop', [8, 8])
@pytest.mark.parametrize('repl', ['z', 'z'])
def test_slice_replace(start, stop, repl):
    s = ["abcdefghij", "0123456789", "9876543210", None, "accénted", ""]
    strs = nvstrings.to_device(s)
    pstrs = pd.Series(s)
    got = strs.slice_replace(start, stop, repl)
    expected = pstrs.str.slice_replace(start, stop, repl)
    assert_eq(got.to_host(), expected)


@pytest.mark.parametrize('index', [0, 3, 9, 10])
def test_get(index):
    index = 0
    s = ["abcdefghij", "0123456789", "9876543210", None, "accénted", ""]
    strs = nvstrings.to_device(s)
    got = strs.get(index)
    expected = ['a', '0', '9', None, 'a', '']
    assert_eq(got.to_host(), expected)


@pytest.mark.parametrize('find', ["3", "3", 'c'])
@pytest.mark.parametrize('replace', ["_", "++", ''])
def test_replace(find, replace):
    s = ["abcdefghij", "0123456789", "9876543210", None, "accénted", ""]
    pstrs = pd.Series(s)
    nvstrs = nvstrings.to_device(s)
    got = nvstrs.replace(find, replace, regex=False)
    expected = pstrs.str.replace(find, replace, regex=False)
    assert_eq(got, expected)


@pytest.mark.parametrize('repl', [''])
def test_fillna(repl):
    s = ["abcdefghij", "0123456789", "9876543210", None, "accénted", ""]
    strs = nvstrings.to_device(s)
    pstrs = pd.Series(s)
    got = strs.fillna(repl)
    expected = pstrs.fillna(repl)
    assert_eq(got.to_host(), expected)
