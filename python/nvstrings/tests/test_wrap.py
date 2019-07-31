# Copyright (c) 2018-2019, NVIDIA CORPORATION.

import pandas as pd
import pytest

import nvstrings
from utils import assert_eq


@pytest.mark.parametrize("width", [10, 20, 50])
def test_wrap(width):
    s = [
        "quick brown fox jumped over lazy brown dog",
        None,
        "hello there, accéntéd world",
        "",
    ]
    strs = nvstrings.to_device(s)
    pstrs = pd.Series(s)
    got = strs.wrap(width)
    expected = pstrs.str.wrap(width)
    assert_eq(got.to_host(), expected)
