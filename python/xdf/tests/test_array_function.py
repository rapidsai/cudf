# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import numpy as np
import pytest

from xdf.autoload import LOADED

if not LOADED:
    import xdf.pandas._testing as tm
else:
    import pandas._testing as tm

from xdf.fast_slow_proxy import make_final_proxy_type


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
        Slow,
        fast_to_slow=lambda fast: Slow(),
        slow_to_fast=lambda slow: Fast(),
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
    )
    with pytest.raises(TypeError):
        np.unique(Proxy())
