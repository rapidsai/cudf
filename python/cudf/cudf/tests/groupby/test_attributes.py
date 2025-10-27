# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
import numpy as np
import pandas as pd
import pytest

import cudf
from cudf.core._compat import (
    PANDAS_CURRENT_SUPPORTED_VERSION,
    PANDAS_VERSION,
)
from cudf.testing import assert_eq


def test_groups():
    # https://github.com/rapidsai/cudf/issues/14955
    df = cudf.DataFrame({"a": [1, 2] * 2}, index=[0] * 4)
    agg = df.groupby("a")
    pagg = df.to_pandas().groupby("a")
    for key in agg.groups:
        np.testing.assert_array_equal(
            pagg.indices[key], agg.indices[key].get()
        )
        assert_eq(pagg.get_group(key), agg.get_group(key))


@pytest.mark.parametrize(
    "by",
    [
        "a",
        "b",
        ["a"],
        ["b"],
        ["a", "b"],
        ["b", "a"],
        np.array([0, 0, 0, 1, 1, 1, 2]),
    ],
)
def test_groupby_groups(by):
    pdf = pd.DataFrame(
        {"a": [1, 2, 1, 2, 1, 2, 3], "b": [1, 2, 3, 4, 5, 6, 7]}
    )
    gdf = cudf.from_pandas(pdf)

    pdg = pdf.groupby(by)
    gdg = gdf.groupby(by)

    for key in pdg.groups:
        assert key in gdg.groups
        assert_eq(pdg.groups[key], gdg.groups[key])


@pytest.mark.parametrize(
    "by",
    [
        "a",
        "b",
        ["a"],
        ["b"],
        ["a", "b"],
        ["b", "a"],
        ["a", "c"],
        ["a", "b", "c"],
    ],
)
def test_groupby_groups_multi(by):
    pdf = pd.DataFrame(
        {
            "a": [1, 2, 1, 2, 1, 2, 3],
            "b": ["a", "b", "a", "b", "b", "c", "c"],
            "c": [1, 2, 3, 4, 5, 6, 7],
        }
    )
    gdf = cudf.from_pandas(pdf)

    pdg = pdf.groupby(by)
    gdg = gdf.groupby(by)

    for key in pdg.groups:
        assert key in gdg.groups
        assert_eq(pdg.groups[key], gdg.groups[key])


def test_groupby_iterate_groups():
    rng = np.random.default_rng(seed=0)
    nelem = 20
    df = cudf.DataFrame(
        {
            "key1": rng.integers(0, 3, nelem),
            "key2": rng.integers(0, 2, nelem),
            "val1": rng.random(nelem),
            "val2": rng.random(nelem),
        }
    )

    def assert_values_equal(arr):
        np.testing.assert_array_equal(arr[0], arr)

    for name, grp in df.groupby(["key1", "key2"]):
        pddf = grp.to_pandas()
        for k in "key1,key2".split(","):
            assert_values_equal(pddf[k].values)


@pytest.mark.parametrize(
    "grouper",
    [
        "a",
        ["a"],
        ["a", "b"],
        np.array([0, 1, 1, 2, 3, 2]),
        {0: "a", 1: "a", 2: "b", 3: "a", 4: "b", 5: "c"},
        lambda x: x + 1,
        ["a", np.array([0, 1, 1, 2, 3, 2])],
    ],
)
def test_grouping(grouper):
    pdf = pd.DataFrame(
        {
            "a": [1, 1, 1, 2, 2, 3],
            "b": [1, 2, 1, 2, 1, 2],
            "c": [1, 2, 3, 4, 5, 6],
        }
    )
    gdf = cudf.from_pandas(pdf)

    for pdf_group, gdf_group in zip(
        pdf.groupby(grouper), gdf.groupby(grouper), strict=True
    ):
        assert pdf_group[0] == gdf_group[0]
        assert_eq(pdf_group[1], gdf_group[1])


@pytest.mark.skipif(
    PANDAS_VERSION < PANDAS_CURRENT_SUPPORTED_VERSION,
    reason="warning not present in older pandas versions",
)
@pytest.mark.parametrize(
    "groups", ["a", "b", "c", ["a", "c"], ["a", "b", "c"]]
)
def test_groupby_dtypes(groups):
    df = cudf.DataFrame(
        {"a": [1, 2, 3, 3], "b": ["x", "y", "z", "a"], "c": [10, 11, 12, 12]}
    )
    pdf = df.to_pandas()
    with pytest.warns(FutureWarning):
        expected = pdf.groupby(groups).dtypes
    with pytest.warns(FutureWarning):
        actual = df.groupby(groups).dtypes

    assert_eq(expected, actual)


def test_ngroups():
    pdf = pd.DataFrame({"a": [1, 1, 3], "b": range(3)})
    gdf = cudf.DataFrame(pdf)

    pgb = pdf.groupby("a")
    ggb = gdf.groupby("a")
    assert pgb.ngroups == ggb.ngroups
    assert len(pgb) == len(ggb)


def test_ndim():
    pdf = pd.DataFrame({"a": [1, 1, 3], "b": range(3)})
    gdf = cudf.DataFrame(pdf)

    pgb = pdf.groupby("a")
    ggb = gdf.groupby("a")
    assert pgb.ndim == ggb.ndim

    pser = pd.Series(range(3))
    gser = cudf.Series(pser)
    pgb = pser.groupby([0, 0, 1])
    ggb = gser.groupby(cudf.Series([0, 0, 1]))
    assert pgb.ndim == ggb.ndim
