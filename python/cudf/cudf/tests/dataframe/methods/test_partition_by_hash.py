# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest

import cudf
from cudf.testing import assert_eq


@pytest.mark.parametrize("nparts", [1, 2])
def test_dataframe_hash_partition(nparts):
    nrows = 10
    nkeys = 2
    rng = np.random.default_rng(seed=0)
    gdf = cudf.DataFrame(
        {f"key{i}": rng.integers(0, 7 - i, nrows) for i in range(nkeys)}
    )
    keycols = gdf.columns.to_list()
    gdf["val1"] = rng.integers(0, nrows * 2, nrows)

    got = gdf.partition_by_hash(keycols, nparts=nparts)
    # Must return a list
    assert isinstance(got, list)
    # Must have correct number of partitions
    assert len(got) == nparts
    # All partitions must be DataFrame type
    assert all(isinstance(p, cudf.DataFrame) for p in got)
    # Check that all partitions have unique keys
    part_unique_keys = set()
    for p in got:
        if len(p):
            # Take rows of the keycolumns and build a set of the key-values
            unique_keys = set(map(tuple, p[keycols].values_host))
            # Ensure that none of the key-values have occurred in other groups
            assert not (unique_keys & part_unique_keys)
            part_unique_keys |= unique_keys
    assert len(part_unique_keys)


def test_dataframe_hash_partition_masked_value():
    nrows = 10
    gdf = cudf.DataFrame(
        {
            "key": np.arange(nrows),
            "val": np.arange(nrows) + 100,
        }
    )
    rng = np.random.default_rng(seed=0)
    boolmask = rng.choice([True, False], size=nrows)
    gdf.loc[boolmask, "val"] = None
    parted = gdf.partition_by_hash(["key"], nparts=3)
    assert len(parted) == 3
    assert_eq(
        parted[0],
        cudf.DataFrame({"key": [0, 8], "val": [100, None]}, index=[0, 8]),
    )
    assert_eq(
        parted[1],
        cudf.DataFrame(
            {"key": range(2, 8), "val": [102] + [None] * 5},
            index=list(range(2, 8)),
        ),
    )
    assert_eq(
        parted[2],
        cudf.DataFrame({"key": [1, 9], "val": [101, 109]}, index=[1, 9]),
    )


def test_dataframe_hash_partition_masked_keys():
    nrows = 5
    gdf = cudf.DataFrame(
        {
            "key": np.arange(nrows),
            "val": np.arange(nrows) + 100,
        }
    )
    rng = np.random.default_rng(seed=0)
    boolmask = rng.choice([True, False], size=nrows)
    gdf.loc[boolmask, "key"] = None
    parted = gdf.partition_by_hash(["key"], nparts=3, keep_index=False)
    assert len(parted) == 3
    assert_eq(parted[0], cudf.DataFrame({"key": [0], "val": [100]}, index=[0]))
    assert_eq(parted[1], cudf.DataFrame({"key": [2], "val": [102]}, index=[0]))
    assert_eq(
        parted[2],
        cudf.DataFrame(
            {"key": [1, None, None], "val": [101, 103, 104]},
            index=list(range(3)),
        ),
    )


@pytest.mark.parametrize("keep_index", [True, False])
def test_dataframe_hash_partition_keep_index(keep_index):
    gdf = cudf.DataFrame(
        {"val": [1, 2, 3, 4, 5], "key": [3, 2, 1, 4, 5]}, index=[5, 4, 3, 2, 1]
    )

    expected_df1 = cudf.DataFrame(
        {"val": [1, 5], "key": [3, 5]}, index=[5, 1] if keep_index else None
    )
    expected_df2 = cudf.DataFrame(
        {"val": [2, 3, 4], "key": [2, 1, 4]},
        index=[4, 3, 2] if keep_index else None,
    )
    expected = [expected_df1, expected_df2]

    parts = gdf.partition_by_hash(["key"], nparts=2, keep_index=keep_index)

    for exp, got in zip(expected, parts, strict=True):
        assert_eq(exp, got)


def test_dataframe_hash_partition_empty():
    gdf = cudf.DataFrame({"val": [1, 2], "key": [3, 2]}, index=["a", "b"])
    parts = gdf.iloc[:0].partition_by_hash(["key"], nparts=3)
    assert len(parts) == 3
    for part in parts:
        assert_eq(gdf.iloc[:0], part)
