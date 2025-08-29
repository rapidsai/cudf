# Copyright (c) 2025, NVIDIA CORPORATION.

import pytest

import cudf
from cudf.testing import assert_eq
from cudf.testing._utils import assert_exceptions_equal


@pytest.mark.parametrize("copy", [True, False])
def test_df_series_dataframe_astype_copy(copy):
    gdf = cudf.DataFrame({"col1": [1, 2], "col2": [3, 4]})
    pdf = gdf.to_pandas()

    assert_eq(
        gdf.astype(dtype="float", copy=copy),
        pdf.astype(dtype="float", copy=copy),
    )
    assert_eq(gdf, pdf)

    gsr = cudf.Series([1, 2])
    psr = gsr.to_pandas()

    assert_eq(
        gsr.astype(dtype="float", copy=copy),
        psr.astype(dtype="float", copy=copy),
    )
    assert_eq(gsr, psr)

    gsr = cudf.Series([1, 2])
    psr = gsr.to_pandas()

    actual = gsr.astype(dtype="int64", copy=copy)
    expected = psr.astype(dtype="int64", copy=copy)
    assert_eq(expected, actual)
    assert_eq(gsr, psr)
    actual[0] = 3
    expected[0] = 3
    assert_eq(gsr, psr)


@pytest.mark.parametrize("copy", [True, False])
def test_df_series_dataframe_astype_dtype_dict(copy):
    gdf = cudf.DataFrame({"col1": [1, 2], "col2": [3, 4]})
    pdf = gdf.to_pandas()

    assert_eq(
        gdf.astype(dtype={"col1": "float"}, copy=copy),
        pdf.astype(dtype={"col1": "float"}, copy=copy),
    )
    assert_eq(gdf, pdf)

    gsr = cudf.Series([1, 2])
    psr = gsr.to_pandas()

    assert_eq(
        gsr.astype(dtype={None: "float"}, copy=copy),
        psr.astype(dtype={None: "float"}, copy=copy),
    )
    assert_eq(gsr, psr)

    assert_exceptions_equal(
        lfunc=psr.astype,
        rfunc=gsr.astype,
        lfunc_args_and_kwargs=([], {"dtype": {"a": "float"}, "copy": copy}),
        rfunc_args_and_kwargs=([], {"dtype": {"a": "float"}, "copy": copy}),
    )

    gsr = cudf.Series([1, 2])
    psr = gsr.to_pandas()

    actual = gsr.astype({None: "int64"}, copy=copy)
    expected = psr.astype({None: "int64"}, copy=copy)
    assert_eq(expected, actual)
    assert_eq(gsr, psr)

    actual[0] = 3
    expected[0] = 3
    assert_eq(gsr, psr)
