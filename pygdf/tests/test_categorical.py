import pytest

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


def test_categorical_compare_unordered():
    cat = pd.Categorical(['a', 'a', 'b', 'c', 'a'], categories=['a', 'b', 'c'])
    pdsr = pd.Series(cat)

    sr = Series.from_any(cat)

    # test equal
    out = sr == sr
    assert out.dtype == np.bool_
    assert type(out[0]) == np.bool_
    assert np.all(out)
    assert np.all(pdsr == pdsr)

    # test inequal
    out = sr != sr
    assert not np.any(out)
    assert not np.any(pdsr != pdsr)

    assert not pdsr.cat.ordered
    assert not sr.cat.ordered

    # test using ordered operators
    with pytest.raises(TypeError) as raises:
        pdsr < pdsr

    raises.match("Unordered Categoricals can only compare equality or not")

    with pytest.raises(TypeError) as raises:
        sr < sr

    raises.match("Unordered Categoricals can only compare equality or not")


def test_categorical_compare_ordered():
    cat1 = pd.Categorical(['a', 'a', 'b', 'c', 'a'],
                          categories=['a', 'b', 'c'], ordered=True)
    pdsr1 = pd.Series(cat1)
    sr1 = Series.from_any(cat1)
    cat2 = pd.Categorical(['a', 'b', 'a', 'c', 'b'],
                          categories=['a', 'b', 'c'], ordered=True)
    pdsr2 = pd.Series(cat2)
    sr2 = Series.from_any(cat2)

    # test equal
    out = sr1 == sr1
    assert out.dtype == np.bool_
    assert type(out[0]) == np.bool_
    assert np.all(out)
    assert np.all(pdsr1 == pdsr1)

    # test inequal
    out = sr1 != sr1
    assert not np.any(out)
    assert not np.any(pdsr1 != pdsr1)

    assert pdsr1.cat.ordered
    assert sr1.cat.ordered

    # test using ordered operators
    np.testing.assert_array_equal(pdsr1 < pdsr2, sr1 < sr2)
    np.testing.assert_array_equal(pdsr1 > pdsr2, sr1 > sr2)


def test_categorical_binary_add():
    cat = pd.Categorical(['a', 'a', 'b', 'c', 'a'], categories=['a', 'b', 'c'])
    pdsr = pd.Series(cat)
    sr = Series.from_any(cat)

    with pytest.raises(TypeError) as raises:
        pdsr + pdsr
    raises.match('Categorical cannot perform the operation \+')

    with pytest.raises(TypeError) as raises:
        sr + sr
    raises.match('Categorical cannot perform the operation: add')


def test_categorical_unary_ceil():
    cat = pd.Categorical(['a', 'a', 'b', 'c', 'a'], categories=['a', 'b', 'c'])
    pdsr = pd.Series(cat)
    sr = Series.from_any(cat)

    with pytest.raises(AttributeError) as raises:
        pdsr.ceil()
    raises.match(r'''no attribute ['"]ceil['"]''')

    with pytest.raises(TypeError) as raises:
        sr.ceil()
    raises.match('Categorical cannot perform the operation: ceil')
