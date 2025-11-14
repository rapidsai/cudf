# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

import pandas as pd
import pytest

import cudf
from cudf.testing import assert_eq
from cudf.testing._utils import assert_exceptions_equal


@pytest.mark.parametrize("name", [None, "old name"])
@pytest.mark.parametrize(
    "names",
    [
        [None, None],
        ["a", None],
        ["new name", "another name"],
        [1, None],
        [2, 3],
        [42, "name"],
    ],
)
@pytest.mark.parametrize("inplace", [True, False])
def test_multiindex_rename(name, names, inplace):
    pi = pd.MultiIndex.from_product(
        [["python", "cobra"], [2018, 2019]], names=[name, None]
    )
    gi = cudf.from_pandas(pi)

    expected = pi.rename(names=names, inplace=inplace)
    actual = gi.rename(names=names, inplace=inplace)

    if inplace:
        expected, actual = pi, gi

    assert_eq(expected, actual)


@pytest.mark.parametrize(
    "names", ["plain string", 123, ["str"], ["l1", "l2", "l3"]]
)
def test_multiindex_rename_error(names):
    pi = pd.MultiIndex.from_product([["python", "cobra"], [2018, 2019]])
    gi = cudf.from_pandas(pi)

    assert_exceptions_equal(
        lfunc=pi.rename,
        rfunc=gi.rename,
        lfunc_args_and_kwargs=([], {"names": names}),
        rfunc_args_and_kwargs=([], {"names": names}),
    )
