import numpy as np
import pytest

import cudf
from cudf.tests.utils import assert_eq, gen_rand_series


def _kernel_multiply(a, b, out):
    for i, (x, y) in enumerate(zip(a, b)):
        out[i] = x * y


def _expect_multiply(a, b, out):
    for i, (x, y) in enumerate(zip(a, b)):
        if x is None or y is None:
            out[i] = None
            return

        if x is np.nan or y is np.nan:
            out[i] = np.nan
            return

        out[i] = x * y


@pytest.mark.parametrize("dtype", ["float32", "float64", "int8", "bool"])
@pytest.mark.parametrize("has_nulls", ["some", "none"])
def test_dataframe_apply_rows(dtype, has_nulls):
    count = 1000
    gdf_series_a = gen_rand_series(dtype, count, has_nulls=has_nulls)
    gdf_series_b = gen_rand_series(dtype, count, has_nulls=has_nulls)
    gdf_series_out = [0] * count

    _expect_multiply(gdf_series_a, gdf_series_b, gdf_series_out)

    df_expected = cudf.DataFrame(
        {"a": gdf_series_a, "b": gdf_series_b, "out": gdf_series_out}
    )

    df_original = cudf.DataFrame({"a": gdf_series_a, "b": gdf_series_b})

    df_actual = df_original.apply_rows(
        _kernel_multiply, ["a", "b"], {"out": dtype}, {}
    )

    assert_eq(df_expected, df_actual)
