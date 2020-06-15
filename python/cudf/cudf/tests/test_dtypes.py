import pandas as pd
import pytest

import cudf
from cudf.core.dtypes import CategoricalDtype
from cudf.tests.utils import assert_eq


def test_cdt_basic():
    psr = pd.Series(["a", "b", "a", "c"], dtype="category")
    sr = cudf.Series(["a", "b", "a", "c"], dtype="category")
    assert isinstance(sr.dtype, CategoricalDtype)
    assert_eq(sr.dtype.categories, psr.dtype.categories)


@pytest.mark.parametrize(
    "data", [None, [], ["a"], [1], [1.0], ["a", "b", "c"]]
)
@pytest.mark.parametrize("ordered", [None, False, True])
def test_cdt_eq(data, ordered):
    dt = cudf.CategoricalDtype(categories=data, ordered=ordered)
    assert dt == "category"
    assert dt == dt
    assert dt == cudf.CategoricalDtype(categories=None, ordered=ordered)
    assert dt == cudf.CategoricalDtype(categories=data, ordered=ordered)
    assert not dt == cudf.CategoricalDtype(
        categories=data, ordered=not ordered
    )


@pytest.mark.parametrize(
    "data", [None, [], ["a"], [1], [1.0], ["a", "b", "c"]]
)
@pytest.mark.parametrize("ordered", [None, False, True])
def test_cdf_to_pandas(data, ordered):
    assert (
        pd.CategoricalDtype(data, ordered)
        == cudf.CategoricalDtype(categories=data, ordered=ordered).to_pandas()
    )


@pytest.mark.parametrize(
    "data",
    [
        {"a": [[]]},
        {"a": [[1, 2, 3]]},
        {"a": [[1, 2, 3]], "b": [2, 3, 4]},
        {"a": [1], "b": [[1, 2, 3]]},
        pd.DataFrame({"a": [[1, 2, 3]]}),
    ],
)
def test_df_list_dtypes(data):

    expectation = pytest.raises(
        NotImplementedError, match="cudf doesn't support list like data types"
    )

    with expectation:
        cudf.DataFrame(data)

    if isinstance(data, pd.DataFrame):
        with expectation:
            cudf.DataFrame.from_pandas(data)


@pytest.mark.parametrize(
    "data", [[[]], [[1, 2, 3], [1, 2, 3]], pd.Series({"a": [[1, 2, 3]]})]
)
def test_sr_list_dtypes(data):

    expectation = pytest.raises(
        NotImplementedError, match="cudf doesn't support list like data types"
    )

    with expectation:
        cudf.Series(data)

    if isinstance(data, pd.Series):
        with expectation:
            cudf.Series.from_pandas(data)
