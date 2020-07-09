import pandas as pd
import pytest

import cudf
from cudf.tests.utils import assert_eq


@pytest.mark.parametrize(
    "data",
    [
        [[]],
        [[[]]],
        [[0]],
        [[0, 1]],
        [[0, 1], [2, 3]],
        [[[0, 1], [2]], [[3, 4]]],
        [[None]],
        [[[None]]],
        [[None], None],
        [[1, None], [1]],
        [[1, None], None],
        [[[1, None], None], None],
    ],
)
def test_create_list_series(data):
    expect = pd.Series(data)
    got = cudf.Series(data)
    assert_eq(expect, got)


@pytest.mark.parametrize(
    "data",
    [
        {"a": [[]]},
        {"a": [[None]]},
        {"a": [[1, 2, 3]]},
        {"a": [[1, 2, 3]], "b": [[2, 3, 4]]},
        {"a": [[1, 2, 3, None], [None]], "b": [[2, 3, 4], [5]], "c": None},
        {"a": [[1]], "b": [[1, 2, 3]]},
        pd.DataFrame({"a": [[1, 2, 3]]}),
    ],
)
def test_df_list_dtypes(data):
    expect = pd.DataFrame(data)
    got = cudf.DataFrame(data)
    assert_eq(expect, got)
