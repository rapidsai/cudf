# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

import pandas as pd
import pytest

import cudf
from cudf.testing import assert_eq
from cudf.testing._utils import assert_exceptions_equal


@pytest.mark.parametrize(
    "idx",
    [
        pd.Index([1, 2, 3]),
        pd.Index(["abc", "def", "ghi"]),
        pd.RangeIndex(0, 10, 1),
        pd.Index([0.324, 0.234, 1.3], name="abc"),
    ],
)
@pytest.mark.parametrize("names", [None, "a", "new name", ["another name"]])
def test_index_set_names(idx, names, inplace):
    if inplace:
        pi = idx.copy()
    else:
        pi = idx
    gi = cudf.from_pandas(idx)

    expected = pi.set_names(names=names, inplace=inplace)
    actual = gi.set_names(names=names, inplace=inplace)

    if inplace:
        expected, actual = pi, gi

    assert_eq(expected, actual)


@pytest.mark.parametrize("level", [1, [0], "abc"])
@pytest.mark.parametrize("names", [None, "a"])
def test_index_set_names_error(level, names):
    pi = pd.Index([1, 2, 3], name="abc")
    gi = cudf.from_pandas(pi)

    assert_exceptions_equal(
        lfunc=pi.set_names,
        rfunc=gi.set_names,
        lfunc_args_and_kwargs=([], {"names": names, "level": level}),
        rfunc_args_and_kwargs=([], {"names": names, "level": level}),
    )
