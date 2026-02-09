# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pandas as pd
import pytest

import cudf
from cudf.testing import assert_eq


@pytest.mark.parametrize("num_id_vars", [0, 2])
@pytest.mark.parametrize("num_value_vars", [0, 2])
@pytest.mark.parametrize("nulls", ["none", "some", "all"])
def test_melt(
    nulls,
    num_id_vars,
    num_value_vars,
    numeric_and_temporal_types_as_str,
    ignore_index,
):
    if numeric_and_temporal_types_as_str not in [
        "float32",
        "float64",
    ] and nulls in ["some", "all"]:
        pytest.skip(
            reason=f"nulls not supported in {numeric_and_temporal_types_as_str}"
        )

    num_rows = 10
    pdf = pd.DataFrame()
    id_vars = []
    rng = np.random.default_rng(seed=0)
    for i in range(num_id_vars):
        colname = "id" + str(i)
        data = rng.integers(0, 26, num_rows).astype(
            numeric_and_temporal_types_as_str
        )
        if nulls == "some":
            idx = rng.choice(num_rows, size=int(num_rows / 2), replace=False)
            data[idx] = np.nan
        elif nulls == "all":
            data[:] = np.nan
        pdf[colname] = data
        id_vars.append(colname)

    value_vars = []
    for i in range(num_value_vars):
        colname = "val" + str(i)
        data = rng.integers(0, 26, num_rows).astype(
            numeric_and_temporal_types_as_str
        )
        if nulls == "some":
            idx = rng.choice(num_rows, size=int(num_rows / 2), replace=False)
            data[idx] = np.nan
        elif nulls == "all":
            data[:] = np.nan
        pdf[colname] = data
        value_vars.append(colname)

    gdf = cudf.from_pandas(pdf)

    got = cudf.melt(
        frame=gdf,
        id_vars=id_vars,
        value_vars=value_vars,
        ignore_index=ignore_index,
    )
    got_from_melt_method = gdf.melt(
        id_vars=id_vars, value_vars=value_vars, ignore_index=ignore_index
    )

    expect = pd.melt(
        frame=pdf,
        id_vars=id_vars,
        value_vars=value_vars,
        ignore_index=ignore_index,
    )

    assert_eq(expect, got)

    assert_eq(expect, got_from_melt_method)


def test_melt_more_than_255_columns():
    mydict = {"id": ["foobar"]}
    for i in range(1, 260):
        mydict[f"d_{i}"] = i

    df = pd.DataFrame(mydict)
    grid_df = pd.melt(df, id_vars=["id"], var_name="d", value_name="sales")

    df_d = cudf.DataFrame(mydict)
    grid_df_d = cudf.melt(
        df_d, id_vars=["id"], var_name="d", value_name="sales"
    )
    grid_df_d["d"] = grid_df_d["d"]

    assert_eq(grid_df, grid_df_d)


def test_melt_str_scalar_id_var():
    data = {"index": [1, 2], "id": [1, 2], "d0": [10, 20], "d1": [30, 40]}
    result = cudf.melt(
        cudf.DataFrame(data),
        id_vars="index",
        var_name="column",
        value_name="value",
    )
    expected = pd.melt(
        pd.DataFrame(data),
        id_vars="index",
        var_name="column",
        value_name="value",
    )
    assert_eq(result, expected)


def test_melt_falsy_var_name():
    df = cudf.DataFrame({"A": ["a", "b", "c"], "B": [1, 3, 5], "C": [2, 4, 6]})
    result = cudf.melt(df, id_vars=["A"], value_vars=["B"], var_name="")
    expected = pd.melt(
        df.to_pandas(), id_vars=["A"], value_vars=["B"], var_name=""
    )
    assert_eq(result, expected)
