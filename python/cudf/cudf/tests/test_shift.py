import numpy as np
import pandas as pd
import pytest

from cudf import DataFrame
from cudf.tests.utils import assert_eq, gen_rand, gen_rand_series


@pytest.mark.parametrize(
    "dtype", ["int8", "int16", "int32", "int64", "float32", "float64"]
)
@pytest.mark.parametrize("periods", [-1, -5, -10, -20, 0, 1, 5, 10, 20])
@pytest.mark.parametrize("has_nulls", [True, False])
def test_shift_series(dtype, periods, has_nulls):
    size = 10000
    if dtype == np.int8:
        # to keep data in range
        data = gen_rand_series(dtype, size, has_nulls=has_nulls, low=-2, high=2)
    else:
        data = gen_rand_series(dtype, size, has_nulls=has_nulls)

    gdf = DataFrame({"a": data})
    pdf = pd.DataFrame({"a": data.to_pandas()})

    shifted_outcome = gdf.a.shift(periods).fillna(-1)
    expected_outcome = pdf.a.shift(periods).fillna(-1).astype(dtype)

    assert_eq(shifted_outcome, expected_outcome)

@pytest.mark.parametrize(
    "dtype", ["int8", "int16", "int32", "int64", "float32", "float64"]
)
@pytest.mark.parametrize("backward", [True, False])
@pytest.mark.parametrize("has_nulls", [True, False])
def test_shift_series_out_of_bounds(dtype, backward, has_nulls):
    size = 10000
    if dtype == np.int8:
        # to keep data in range
        data = gen_rand_series(dtype, size, has_nulls=has_nulls, low=-2, high=2)
    else:
        data = gen_rand_series(dtype, size, has_nulls=has_nulls)

    gdf = DataFrame({"a": data})
    pdf = pd.DataFrame({"a": data.to_pandas()})

    periods = size + 1 if not backward else -size - 1

    shifted_outcome = gdf.a.shift(periods).fillna(-1)
    expected_outcome = pdf.a.shift(periods).fillna(-1).astype(dtype)

    assert_eq(shifted_outcome, expected_outcome)

@pytest.mark.parametrize(
    "dtype", ["int8", "int16", "int32", "int64", "float32", "float64"]
)
@pytest.mark.parametrize("has_nulls", [True, False])
def test_shift_series_zero(dtype, has_nulls):
    size = 10000
    periods = 0
    if dtype == np.int8:
        # to keep data in range
        data = gen_rand_series(dtype, size, has_nulls=has_nulls, low=-2, high=2)
    else:
        data = gen_rand_series(dtype, size, has_nulls=has_nulls)

    gdf = DataFrame({"a": data})
    pdf = pd.DataFrame({"a": data.to_pandas()})

    shifted_outcome = gdf.a.shift(periods).fillna(-1)
    expected_outcome = pdf.a.shift(periods).fillna(-1).astype(dtype)

    assert_eq(shifted_outcome, expected_outcome)

def test_shift_dataframe():
    source = DataFrame({
        'x': [0., 1., 2., None, 4., 5.],
        'y': [5, 4, 3, None, 1, 0]
    })
    expected = DataFrame({
        'x': [None, None, 0., 1., 2., None],
        'y': [None, None, 5, 4, 3, None]
    })

    actual = source.shift(2)

    assert_eq(expected, actual)
