# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

import pandas as pd
import pytest

from cudf import Index


@pytest.mark.parametrize(
    "testlist",
    [
        [10, 9, 8, 8, 7],
        [2.0, 5.0, 4.0, 3.0, 7.0],
        ["b", "cat", "e", "bat", "c"],
    ],
)
@pytest.mark.parametrize("side", ["left", "right"])
def test_get_slice_bound(testlist, side):
    index = Index(testlist)
    index_pd = pd.Index(testlist)
    for label in testlist:
        expect = index_pd.get_slice_bound(label, side)
        got = index.get_slice_bound(label, side)
        assert got == expect


@pytest.mark.parametrize("label", [1, 5, 7, 11])
@pytest.mark.parametrize("side", ["left", "right"])
def test_get_slice_bound_missing(label, side):
    mylist = [2, 4, 6, 8, 10]
    index = Index(mylist)
    index_pd = pd.Index(mylist)

    expect = index_pd.get_slice_bound(label, side)
    got = index.get_slice_bound(label, side)
    assert got == expect


@pytest.mark.parametrize("label", ["a", "c", "g"])
@pytest.mark.parametrize("side", ["left", "right"])
def test_get_slice_bound_missing_str(label, side):
    mylist = ["b", "d", "f"]
    index = Index(mylist)
    index_pd = pd.Index(mylist)
    got = index.get_slice_bound(label, side)
    expect = index_pd.get_slice_bound(label, side)
    assert got == expect
