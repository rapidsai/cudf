# Copyright (c) 2019, NVIDIA CORPORATION.

import pytest
import pandas as pd
import numpy as np

import cudf
from hypothesis import given, settings
import hypothesis.strategies as st


@given(st.lists(st.integers(-9223372036854775808, 9223372036854775807)))
@settings(deadline=None)
def test_integer_dataframe(x):
    if not hasattr(pd, 'Int64Dtype'):
        pytest.skip(msg="Test only supported with Pandas >= 0.24.2")
    gdf = cudf.DataFrame({'x': x})
    pdf = gdf.to_pandas()
    assert gdf.__repr__() == pdf.__repr__()


@given(st.lists(st.integers(-9223372036854775808, 9223372036854775807)))
@settings(deadline=None)
def test_integer_series(x):
    sr = cudf.Series(x)
    ps = pd.Series(x)
    print(sr)
    print(ps)
    assert sr.__repr__() == ps.__repr__()


@given(st.lists(st.floats()))
def test_float_dataframe(x):
    if not hasattr(pd, 'Int64Dtype'):
        pytest.skip(msg="Test only supported with Pandas >= 0.24.2")
    gdf = cudf.DataFrame({'x': cudf.Series(x, nan_as_null=False)})
    pdf = gdf.to_pandas()
    assert gdf.__repr__() == pdf.__repr__()


@given(st.lists(st.floats()))
def test_float_series(x):
    sr = cudf.Series(x, nan_as_null=False)
    ps = pd.Series(x)
    assert sr.__repr__() == ps.__repr__()


def test_mixed_dataframe():
    if not hasattr(pd, 'Int64Dtype'):
        pytest.skip(msg="Test only supported with Pandas >= 0.24.2")
    pdf = pd.DataFrame()
    pdf['Integer'] = np.array([2345, 11987, 9027, 9027])
    pdf['Date'] = np.array(['18/04/1995', '14/07/1994', '07/06/2006',
                           '16/09/2005'])
    pdf['Float'] = np.array([9.001, 8.343, 6, 2.781])
    pdf['Integer2'] = np.array([2345, 106, 2088, 789277])
    pdf['Category'] = np.array(['M', 'F', 'F', 'F'])
    pdf['String'] = np.array(['Alpha', 'Beta', 'Gamma', 'Delta'])
    pdf['Boolean'] = np.array([True, False, True, False])
    gdf = cudf.from_pandas(pdf)
    assert gdf.__repr__() == pdf.__repr__()


def test_MI():
    if not hasattr(pd, 'Int64Dtype'):
        pytest.skip(msg="Test only supported with Pandas >= 0.24.2")
    gdf = cudf.DataFrame({'a': np.random.randint(0, 4, 10),
                          'b': np.random.randint(0, 4, 10),
                          'c': np.random.randint(0, 4, 10)})
    levels = [['a', 'b', 'c', 'd'], ['w', 'x', 'y', 'z'], ['m', 'n']]
    codes = cudf.DataFrame({'a': np.random.randint(0, 4, 10),
                            'b': np.random.randint(0, 4, 10),
                            'c': np.random.randint(0, 2, 10)})
    pd.options.display.max_rows = 999
    gdf = gdf.set_index(cudf.MultiIndex(levels=levels, codes=codes))
    pdf = gdf.to_pandas()
    gdfT = gdf.T
    pdfT = pdf.T
    assert gdf.__repr__() == pdf.__repr__()
    assert gdfT.__repr__() == pdfT.__repr__()


@pytest.mark.parametrize('nrows', [0, 1, 3, 5, 10])
@pytest.mark.parametrize('ncols', [0, 1, 2, 3])
def test_groupby_MI(nrows, ncols):
    if not hasattr(pd, 'Int64Dtype'):
        pytest.skip(msg="Test only supported with Pandas >= 0.24.2")
    gdf = cudf.DataFrame({'a': np.random.randint(0, 4, 10),
                          'b': np.random.randint(0, 4, 10),
                          'c': np.random.randint(0, 4, 10)})
    pdf = gdf.to_pandas()
    gdg = gdf.groupby(['a', 'b']).count()
    pdg = pdf.groupby(['a', 'b']).count()
    pd.options.display.max_rows = nrows
    pd.options.display.max_columns = ncols
    assert gdg.__repr__() == pdg.__repr__()
    assert gdg.T.__repr__() == pdg.T.__repr__()
