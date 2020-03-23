import pandas as pd
import pytest

import cudf
from cudf.tests.utils import assert_eq

quantile_test_values = [0, 1, [], [0.5], [0.5, 0, 1]]
interpolation_test_values = [
    "linear",
    "midpoint",
    "higher",
    "lower",
    "nearest",
]


def test_dataframe_no_columns():
    with pytest.raises(ValueError):
        cudf.DataFrame().quantile()

    with pytest.raises(ValueError):
        pd.DataFrame().quantile()

    with pytest.raises(ValueError):
        cudf.DataFrame().quantile([0])

    with pytest.raises(ValueError):
        pd.DataFrame().quantile([0])


@pytest.mark.parametrize("q", quantile_test_values)
@pytest.mark.parametrize("interp", interpolation_test_values)
@pytest.mark.parametrize(
    "data", [{"x": []}, {"x": [], "y": []}, {"x": [0, 2, 1], "y": [1, 0, 2]}]
)
def test_dataframe(q, interp, data):
    expected = pd.DataFrame(data).quantile(q, interpolation=interp)
    actual = cudf.DataFrame(data).quantile(q, interpolation=interp)
    assert_eq(expected, actual)


@pytest.mark.parametrize("q", quantile_test_values)
@pytest.mark.parametrize("interp", interpolation_test_values)
@pytest.mark.parametrize("data", [[], [3, 2, 1], [1, 2, 3]])
@pytest.mark.parametrize("name", ["x", None])
def test_series(q, interp, data, name):
    expected = pd.Series(data, name=name).quantile(q, interpolation=interp)
    actual = cudf.Series(data, name=name).quantile(q, interpolation=interp)
    assert_eq(expected, actual)
