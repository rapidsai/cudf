import pandas as pd
import pytest

import cudf
from cudf.tests.utils import assert_eq

interpolation_test_values = [
    "linear",
    "midpoint",
    "higher",
    "lower",
    "nearest",
]
quantile_test_values = [0, 1, [], [0.5, 0, 1]]


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
    print(expected)
    print(actual)
    assert_eq(expected, actual)


@pytest.mark.parametrize("q", quantile_test_values)
@pytest.mark.parametrize("interp", interpolation_test_values)
@pytest.mark.parametrize("data", [[], [3, 2, 1], [1, 2, 3]])
def test_series(q, interp, data):
    expected = pd.Series(data).quantile(q, interpolation=interp)
    actual = cudf.Series(data).quantile(q, interpolation=interp)
    assert_eq(expected, actual)
