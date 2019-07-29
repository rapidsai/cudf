# Copyright (c) 2018-2019, NVIDIA CORPORATION.

import nvstrings
import pytest
from utils import methodcaller


@pytest.mark.parametrize('func',
                         ['lower', 'upper',
                          'swapcase', 'capitalize',
                          'title', 'strip'])
def test_allnulls(func):
    strs = nvstrings.to_device([None, None, None])
    M = methodcaller(func)

    assert M(strs).to_host() == [None, None, None]
