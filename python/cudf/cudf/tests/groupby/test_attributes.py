# Copyright (c) 2024-2025, NVIDIA CORPORATION.
import numpy as np
import pandas as pd
import pytest

import cudf
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
