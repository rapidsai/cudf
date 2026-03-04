# SPDX-FileCopyrightText: Copyright (c) 2023-2025, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest

from cudf.pandas import LOADED
from cudf.pandas._wrappers.common import array_function_method

if not LOADED:
    raise ImportError("These tests must be run with cudf.pandas loaded")

import pandas._testing as tm

from cudf.pandas.fast_slow_proxy import make_final_proxy_type


class Slow:
    def __array__(self):
        return np.array([1, 1, 1, 2, 2, 3])


class Slow2:
    def __array_function__(self, func, types, args, kwargs):
        return "slow"


class Fast:
    def __array_function__(self, func, types, args, kwargs):
        return "fast"


class Fast2:
    def __array_function__(self, func, types, args, kwargs):
        return NotImplemented


def test_array_function():
    # test that fast dispatch to __array_function__ works
    Proxy = make_final_proxy_type(
        "Proxy",
        Fast,
        Slow2,
        fast_to_slow=lambda fast: Slow2(),
        slow_to_fast=lambda slow: Fast(),
        additional_attributes={"__array_function__": array_function_method},
    )
    tm.assert_equal(np.unique(Proxy()), "fast")


def test_array_function_fallback():
    # test that slow dispatch works when the fast dispatch fails
    Proxy = make_final_proxy_type(
        "Proxy",
        Fast2,
        Slow2,
        fast_to_slow=lambda fast: Slow2(),
        slow_to_fast=lambda slow: Fast2(),
        additional_attributes={"__array_function__": array_function_method},
    )
    tm.assert_equal(np.unique(Proxy()), "slow")


def test_array_function_fallback_array():
    # test that dispatch to slow __array__ works when
    # fast __array_function__ fails
    Proxy = make_final_proxy_type(
        "Proxy",
        Fast2,
        Slow,
        fast_to_slow=lambda fast: Slow(),
        slow_to_fast=lambda slow: Fast2(),
        additional_attributes={"__array_function__": array_function_method},
    )
    tm.assert_equal(np.unique(Proxy()), np.unique(np.asarray(Slow())))


def test_array_function_notimplemented():
    # tests that when neither Fast nor Slow implement __array_function__,
    # we get a TypeError
    Proxy = make_final_proxy_type(
        "Proxy",
        Fast2,
        Fast2,
        fast_to_slow=lambda fast: Fast2(),
        slow_to_fast=lambda slow: Fast2(),
        additional_attributes={"__array_function__": array_function_method},
    )
    with pytest.raises(TypeError):
        np.unique(Proxy())
