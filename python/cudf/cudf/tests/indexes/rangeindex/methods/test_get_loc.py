# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

import pandas as pd
import pytest

import cudf
from cudf.testing import assert_eq
from cudf.testing._utils import assert_exceptions_equal


@pytest.mark.parametrize("key", list(range(1, 110, 13)))
def test_get_loc_rangeindex(key):
    pi = pd.RangeIndex(3, 100, 4)
    gi = cudf.from_pandas(pi)
    if (
        (key not in pi)
        # Get key before the first element is KeyError
        or (key < pi.start)
        # Get key after the last element is KeyError
        or (key >= pi.stop)
    ):
        assert_exceptions_equal(
            lfunc=pi.get_loc,
            rfunc=gi.get_loc,
            lfunc_args_and_kwargs=([], {"key": key}),
            rfunc_args_and_kwargs=([], {"key": key}),
        )
    else:
        expected = pi.get_loc(key)
        got = gi.get_loc(key)

        assert_eq(expected, got)
