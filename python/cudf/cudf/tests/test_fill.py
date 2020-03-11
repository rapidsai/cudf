import pytest
from pandas.util.testing import assert_series_equal

import cudf


def cpu_fill(data, fill_value, begin, end):

    if end == -1:
        end = len(data)

    fill_size = end - begin

    a = list(data[i] for i in range(begin))
    b = list(fill_value for i in range(fill_size))
    c = list(data[i + begin + fill_size] for i in range(len(data) - end))

    return a + b + c


@pytest.mark.parametrize(
    "fill_value,data",
    [
        ("x", ["a", "b", "c", "d", "e", "f"]),
        (7, [6, 3, 4, 2, 1, 7, 8, 5]),
        (0.8, [0.6, 0.3, 0.4, 0.2, 0.1, 0.7, 0.8, 0.5]),
    ],
)
@pytest.mark.parametrize("begin,end", [(0, -1), (0, 4), (1, -1), (1, 4)])
def test_fill(data, fill_value, begin, end):
    original = cudf.Series(data)
    expected = cudf.Series(cpu_fill(data, fill_value, begin, end))

    actual = original.fill(fill_value, begin, end)

    print(cpu_fill(data, fill_value, begin, end))
    print(actual)
    print(expected)

    assert_series_equal(expected.to_pandas(), actual.to_pandas())
