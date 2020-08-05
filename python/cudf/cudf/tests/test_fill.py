import pandas as pd
import pytest

import cudf
from cudf.tests.utils import assert_eq


@pytest.mark.parametrize(
    "fill_value,data",
    [
        (7, [6, 3, 4]),
        ("x", ["a", "b", "c", "d", "e", "f"]),
        (7, [6, 3, 4, 2, 1, 7, 8, 5]),
        (0.8, [0.6, 0.3, 0.4, 0.2, 0.1, 0.7, 0.8, 0.5]),
        ("b", pd.Categorical(["a", "b", "c"])),
        (None, [0.0, 1.0, 2.0, 3.0]),
    ],
)
@pytest.mark.parametrize(
    "begin,end",
    [
        (0, -1),
        (0, 4),
        (1, -1),
        (1, 4),
        (-2, 1),
        (-2, -1),
        (10, 12),
        (8, 10),
        (10, 8),
        (-10, -8),
        (-2, 6),
    ],
)
@pytest.mark.parametrize("inplace", [True, False])
def test_fill(data, fill_value, begin, end, inplace):
    gs = cudf.Series(data)
    ps = gs.to_pandas(nullable_pd_dtype=False)

    if inplace:
        actual = gs
        gs[begin:end] = fill_value
    else:
        # private impl doesn't take care of rounding or bounds check
        if begin < 0:
            begin += len(gs)

        if end < 0:
            end += len(gs)

        begin = max(0, min(len(gs), begin))
        end = max(0, min(len(gs), end))
        actual = gs._fill([fill_value], begin, end, False)
        assert actual is not gs

    ps[begin:end] = fill_value

    assert_eq(ps, actual)


@pytest.mark.xfail(raises=ValueError)
def test_fill_new_category():
    gs = cudf.Series(pd.Categorical(["a", "b", "c"]))
    gs[0:1] = "d"
