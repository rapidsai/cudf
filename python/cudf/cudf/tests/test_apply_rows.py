# Copyright (c) 2019-2024, NVIDIA CORPORATION.

import pytest

import cudf
from cudf.core.column import column
from cudf.testing import assert_eq
from cudf.testing._utils import gen_rand_series


def _kernel_multiply(a, b, out):
    for i, (x, y) in enumerate(zip(a, b)):
        out[i] = x * y


@pytest.mark.parametrize("dtype", ["float32", "float64"])
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
        a = cudf.Series._from_data(
            {None: column.build_column(gdf_series_a.data, dtype)}
        )
        b = cudf.Series_from_data(
            {None: column.build_column(gdf_series_b.data, dtype)}
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

    df_actual = df_original.apply_rows(
        _kernel_multiply,
        ["a", "b"],
        {"out": dtype},
        {},
        pessimistic_nulls=pessimistic,
    )

    assert_eq(df_expected, df_actual)
