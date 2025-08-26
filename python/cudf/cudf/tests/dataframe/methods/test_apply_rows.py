# Copyright (c) 2025, NVIDIA CORPORATION.

import numpy as np
import pytest

import cudf
from cudf import DataFrame
from cudf.core.column import column
from cudf.testing import assert_eq
from cudf.testing._utils import gen_rand_series


def _kernel_multiply(a, b, out):
    # numba doesn't support zip(..., strict=True), so we must tell ruff to ignore it.
    for i, (x, y) in enumerate(zip(a, b)):  # noqa: B905
        out[i] = x * y


@pytest.mark.parametrize("dtype", [np.dtype("float32"), np.dtype("float64")])
@pytest.mark.parametrize("has_nulls", [False, True])
@pytest.mark.parametrize("pessimistic", [False, True])
def test_dataframe_apply_rows(dtype, has_nulls, pessimistic):
    count = 1000
    gdf_series_a = gen_rand_series(dtype, count, has_nulls=has_nulls)
    gdf_series_b = gen_rand_series(dtype, count, has_nulls=has_nulls)
    gdf_series_c = gen_rand_series(dtype, count, has_nulls=has_nulls)

    if pessimistic:
        # pessimistically combine the null masks
        gdf_series_expected = gdf_series_a * gdf_series_b
    else:
        # optimistically ignore the null masks
        a = cudf.Series._from_column(
            column.build_column(gdf_series_a.data, dtype)
        )
        b = cudf.Series._from_column(
            column.build_column(gdf_series_b.data, dtype)
        )
        gdf_series_expected = a * b

    df_expected = cudf.DataFrame(
        {
            "a": gdf_series_a,
            "b": gdf_series_b,
            "c": gdf_series_c,
            "out": gdf_series_expected,
        }
    )

    df_original = cudf.DataFrame(
        {"a": gdf_series_a, "b": gdf_series_b, "c": gdf_series_c}
    )

    with pytest.warns(FutureWarning):
        df_actual = df_original.apply_rows(
            _kernel_multiply,
            ["a", "b"],
            {"out": dtype},
            {},
            pessimistic_nulls=pessimistic,
        )

    assert_eq(df_expected, df_actual)


def test_df_apply_rows():
    nelem = 20

    def kernel(in1, in2, in3, out1, out2, extra1, extra2):
        for i, (x, y, z) in enumerate(zip(in1, in2, in3)):  # noqa: B905
            out1[i] = extra2 * x - extra1 * y
            out2[i] = y - extra1 * z

    in1 = np.arange(nelem)
    in2 = np.arange(nelem)
    in3 = np.arange(nelem)

    df = DataFrame(
        {
            "in1": in1,
            "in2": in2,
            "in3": in3,
        }
    )

    extra1 = 2.3
    extra2 = 3.4

    expect_out1 = extra2 * in1 - extra1 * in2
    expect_out2 = in2 - extra1 * in3

    with pytest.warns(FutureWarning):
        outdf = df.apply_rows(
            kernel,
            incols=["in1", "in2", "in3"],
            outcols=dict(out1=np.float64, out2=np.float64),
            kwargs=dict(extra1=extra1, extra2=extra2),
        )

    got_out1 = outdf["out1"].to_numpy()
    got_out2 = outdf["out2"].to_numpy()

    np.testing.assert_array_almost_equal(got_out1, expect_out1)
    np.testing.assert_array_almost_equal(got_out2, expect_out2)


def test_df_apply_rows_incols_mapping():
    nelem = 20

    def kernel(x, y, z, out1, out2, extra1, extra2):
        for i, (a, b, c) in enumerate(zip(x, y, z)):  # noqa: B905
            out1[i] = extra2 * a - extra1 * b
            out2[i] = b - extra1 * c

    df = DataFrame()
    df["in1"] = in1 = np.arange(nelem)
    df["in2"] = in2 = np.arange(nelem)
    df["in3"] = in3 = np.arange(nelem)

    extra1 = 2.3
    extra2 = 3.4

    expected_out = DataFrame()
    expected_out["out1"] = extra2 * in1 - extra1 * in2
    expected_out["out2"] = in2 - extra1 * in3

    with pytest.warns(FutureWarning):
        outdf = df.apply_rows(
            kernel,
            incols={"in1": "x", "in2": "y", "in3": "z"},
            outcols=dict(out1=np.float64, out2=np.float64),
            kwargs=dict(extra1=extra1, extra2=extra2),
        )

    assert_eq(outdf[["out1", "out2"]], expected_out)
