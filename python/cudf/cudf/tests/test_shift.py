import numpy as np
import pandas as pd
import pytest

from cudf import DataFrame
from cudf.tests.utils import assert_eq, gen_rand


@pytest.mark.parametrize(
    "dtype", ["int8", "int16", "int32", "int64", "float32", "float64"]
)
@pytest.mark.parametrize("periods", [-1, -5, -10, -20, 0, 1, 5, 10, 20])
def test_shift_series(dtype, periods):
    if dtype == np.int8:
        # to keep data in range
        data = gen_rand(dtype, 100000, low=-2, high=2)
    else:
        data = gen_rand(dtype, 100000)

    gdf = DataFrame({"a": data})
    pdf = pd.DataFrame({"a": data})

    shifted_outcome = gdf.a.shift(periods)
    expected_outcome = pdf.a.shift(periods).fillna(-1).astype(dtype)

    assert_eq(shifted_outcome, expected_outcome)

def test_shift_dataframe():
    source = DataFrame({
        'x': [0., 1., 2., None, .4, .5],
        'y': [5, 4, 3, None, 1, 0]
    })
    expected = DataFrame({
        'x': [None, None, 0., 1., 2., None],
        'y': [None, None, 5, 4, 3, None]
    })

    actual = source.shift(2)

    print(source)
    print(expected)
    print(actual.x.has_null_mask)

    assert_eq(expected, actual)
