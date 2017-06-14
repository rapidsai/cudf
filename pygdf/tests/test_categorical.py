import numpy as np
import pandas as pd

from pygdf.dataframe import Series


def test_categorical_basic():
    cat = pd.Categorical(['a', 'a', 'b', 'c', 'a'], categories=['a', 'b', 'c'])
    pdsr = pd.Series(cat)
    sr = Series.from_any(cat)
    np.testing.assert_array_equal(cat.codes, sr.to_array())
    assert sr.dtype == pdsr.dtype

    # Test attributes
    assert tuple(pdsr.cat.categories) == tuple(sr.cat.categories)
    assert pdsr.cat.ordered == sr.cat.ordered

    np.testing.assert_array_equal(pdsr.cat.codes.data, sr.cat.codes.to_array())
    np.testing.assert_array_equal(pdsr.cat.codes.dtype, sr.cat.codes.dtype)

    string = str(sr)
    expect_str = """
0 a
1 a
2 b
3 c
4 a
"""
    assert all(x == y for x, y in zip(string.split(), expect_str.split()))


def test_categorical_missing():
    cat = pd.Categorical(['a', '_', '_', 'c', 'a'], categories=['a', 'b', 'c'])
    pdsr = pd.Series(cat)
    sr = Series.from_any(cat)
    np.testing.assert_array_equal(cat.codes, sr.to_array(fillna='pandas'))
    assert sr.null_count == 2

    np.testing.assert_array_equal(pdsr.cat.codes.data,
                                  sr.cat.codes.to_array(fillna='pandas'))
    np.testing.assert_array_equal(pdsr.cat.codes.dtype, sr.cat.codes.dtype)

    string = str(sr)
    expect_str = """
0 a
1
2
3 c
4 a
"""
    assert string.split() == expect_str.split()
