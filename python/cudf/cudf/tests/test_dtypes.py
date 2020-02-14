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


@pytest.mark.parametrize(
    "data", [None, [], ["a"], [1], [1.0], ["a", "b", "c"]]
)
@pytest.mark.parametrize("ordered", [None, False, True])
def test_cdf_to_pandas(data, ordered):
    assert (
        pd.CategoricalDtype(data, ordered)
        == cudf.CategoricalDtype(categories=data, ordered=ordered).to_pandas()
    )
