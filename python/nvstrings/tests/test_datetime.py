# Copyright (c) 2018-2019, NVIDIA CORPORATION.

import numpy as np
import pandas as pd

import nvstrings

from utils import assert_eq


def test_timestamp2int():
    s = nvstrings.to_device(["2019-03-20T12:34:56Z", "2020-02-29T23:59:59Z"])
    s1 = pd.Series(["2019-03-20T12:34:56Z", "2020-02-29T23:59:59Z"]).apply(
        lambda x: pd.Timestamp(x))
    got = s.timestamp2int()
    expected = s1.astype(np.int64)
    assert np.allclose(got, expected, 10)

    s1 = pd.Series(["2019-03-20T12:34:56Z", "2020-02-29T23:59:59Z"]).apply(
        lambda x: pd.Timestamp(x, unit='ms'))
    got = s.timestamp2int(units='ms')
    expected = s1.astype(np.int64)
    assert np.allclose(got, expected, 10)


def test_int2timestamp():
    ints = [1553085296, 1582934400]
    got = nvstrings.int2timestamp(ints)
    expected = ['2019-03-20T12:34:56Z', '2020-02-29T00:00:00Z']
    assert_eq(got, expected)
