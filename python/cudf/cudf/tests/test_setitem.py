# Copyright (c) 2018-2020, NVIDIA CORPORATION.

import numpy as np
import pandas as pd
import pytest

import cudf
from cudf.tests.utils import assert_eq


@pytest.mark.parametrize("df", [pd.DataFrame({"a": [1, 2, 3]})])
@pytest.mark.parametrize("arg", [[True, False, True], [True, True, True]])
@pytest.mark.parametrize("value", [0, -1])
def test_dataframe_setitem_bool_mask_scaler(df, arg, value):
    gdf = cudf.from_pandas(df)

    df[arg] = value
    gdf[arg] = value
    assert_eq(df, gdf)


# pandas incorrectly adds nulls with dataframes
# but works fine with scalers
@pytest.mark.xfail()
def test_dataframe_setitem_scaler_bool_inconsistency():
    df = pd.DataFrame({"a": [1, 2, 3]})
    df[[True, False, True]] = pd.DataFrame({"a": [-1, -2]})

    gdf = cudf.DataFrame({"a": [1, 2, 3]})
    gdf[[True, False, True]] = cudf.DataFrame({"a": [-1, -2]})
    assert_eq(df, gdf)


@pytest.mark.parametrize("df", [pd.DataFrame({"a": [1, 2, 3]})])
@pytest.mark.parametrize("arg", [["a"], "a", "b"])
@pytest.mark.parametrize("value", [-10, pd.DataFrame({"a": [-1, -2, -3]})])
def test_dataframe_setitem_columns(df, arg, value):
    gdf = cudf.from_pandas(df)
    cudf_replace_value = value

    if isinstance(cudf_replace_value, pd.DataFrame):
        cudf_replace_value = cudf.from_pandas(value)

    df[arg] = value
    gdf[arg] = cudf_replace_value
    assert_eq(df, gdf, check_dtype=False)


@pytest.mark.parametrize("df", [pd.DataFrame({"a": [1, 2, 3]})])
@pytest.mark.parametrize("arg", [["b", "c"]])
@pytest.mark.parametrize(
    "value",
    [
        pd.DataFrame({"0": [-1, -2, -3], "1": [-0, -10, -1]}),
        10,
        20,
        30,
        "rapids",
        "ai",
        0.32234,
        np.datetime64(1324232423423342, "ns"),
        np.timedelta64(34234324234324234, "ns"),
    ],
)
def test_dataframe_setitem_new_columns(df, arg, value):
    gdf = cudf.from_pandas(df)
    cudf_replace_value = value

    if isinstance(cudf_replace_value, pd.DataFrame):
        cudf_replace_value = cudf.from_pandas(value)

    df[arg] = value
    gdf[arg] = cudf_replace_value
    assert_eq(df, gdf, check_dtype=True)


# set_item_series inconsistency
def test_series_setitem_index():
    df = pd.DataFrame(
        data={"b": [-1, -2, -3], "c": [1, 2, 3]}, index=[1, 2, 3]
    )

    df["b"] = pd.Series(data=[12, 11, 10], index=[3, 2, 1])
    gdf = cudf.DataFrame(
        data={"b": [-1, -2, -3], "c": [1, 2, 3]}, index=[1, 2, 3]
    )
    gdf["b"] = cudf.Series(data=[12, 11, 10], index=[3, 2, 1])
    assert_eq(df, gdf, check_dtype=False)


@pytest.mark.parametrize("psr", [pd.Series([1, 2, 3], index=["a", "b", "c"])])
@pytest.mark.parametrize(
    "arg", ["b", ["a", "c"], slice(1, 2, 1), [True, False, True]]
)
def test_series_set_item(psr, arg):
    gsr = cudf.from_pandas(psr)

    psr[arg] = 11
    gsr[arg] = 11

    assert_eq(psr, gsr)


@pytest.mark.parametrize(
    "df",
    [
        pd.DataFrame(
            {"a": [1, 2, 3]},
            index=pd.MultiIndex.from_frame(
                pd.DataFrame({"b": [3, 2, 1], "c": ["a", "b", "c"]})
            ),
        ),
        pd.DataFrame({"a": [1, 2, 3]}, index=["a", "b", "c"]),
    ],
)
def test_setitem_dataframe_series_inplace(df):
    pdf = df
    gdf = cudf.from_pandas(pdf)

    pdf["a"].replace(1, 500, inplace=True)
    gdf["a"].replace(1, 500, inplace=True)

    assert_eq(pdf, gdf)

    psr_a = pdf["a"]
    gsr_a = gdf["a"]

    psr_a.replace(500, 501, inplace=True)
    gsr_a.replace(500, 501, inplace=True)

    assert_eq(pdf, gdf)


@pytest.mark.parametrize(
    "replace_data",
    [
        [100, 200, 300, 400, 500],
        cudf.Series([100, 200, 300, 400, 500]),
        cudf.Series([100, 200, 300, 400, 500], index=[2, 3, 4, 5, 6]),
    ],
)
def test_series_set_equal_length_object_by_mask(replace_data):

    psr = pd.Series([1, 2, 3, 4, 5])
    gsr = cudf.from_pandas(psr)

    # Lengths match in trivial case
    pd_bool_col = pd.Series([True] * len(psr))
    gd_bool_col = cudf.from_pandas(pd_bool_col)

    psr[pd_bool_col] = (
        replace_data.to_pandas()
        if hasattr(replace_data, "to_pandas")
        else replace_data
    )
    gsr[gd_bool_col] = replace_data

    assert_eq(psr.astype("float"), gsr.astype("float"))

    # Test partial masking
    psr[psr > 1] = (
        replace_data.to_pandas()
        if hasattr(replace_data, "to_pandas")
        else replace_data
    )
    gsr[gsr > 1] = replace_data

    assert_eq(psr.astype("float"), gsr.astype("float"))


def test_column_set_equal_length_object_by_mask():
    # Series.__setitem__ might bypass some of the cases
    # handled in column.__setitem__ so this test is needed

    data = cudf.Series([0, 0, 1, 1, 1])._column
    replace_data = cudf.Series([100, 200, 300, 400, 500])._column
    bool_col = cudf.Series([True, True, True, True, True])._column

    data[bool_col] = replace_data
    assert_eq(cudf.Series(data), cudf.Series(replace_data))

    data = cudf.Series([0, 0, 1, 1, 1])._column
    bool_col = cudf.Series([True, False, True, False, True])._column
    data[bool_col] = replace_data

    assert_eq(cudf.Series(data), cudf.Series([100, 0, 300, 1, 500]))
