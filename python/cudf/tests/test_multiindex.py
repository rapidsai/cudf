# Copyright (c) 2019, NVIDIA CORPORATION.

"""
Test related to MultiIndex
"""
import pytest

import cudf
import numpy as np
import pandas as pd

from cudf.tests.utils import assert_eq


def test_multiindex_levels_codes_validation():
    levels = [['a', 'b'], ['c', 'd']]
    # Codes not a sequence of sequences
    with pytest.raises(TypeError):
        pd.MultiIndex(levels, [0, 1])
    with pytest.raises(TypeError):
        cudf.MultiIndex(levels, [0, 1])
    # Codes don't match levels
    with pytest.raises(ValueError):
        pd.MultiIndex(levels, [[0], [1], [1]])
    with pytest.raises(ValueError):
        cudf.MultiIndex(levels, [[0], [1], [1]])
    # Largest code greater than number of levels
    with pytest.raises(ValueError):
        pd.MultiIndex(levels, [[0, 1], [0, 2]])
    with pytest.raises(ValueError):
        cudf.MultiIndex(levels, [[0, 1], [0, 2]])
    # Unequal code lengths
    with pytest.raises(ValueError):
        pd.MultiIndex(levels, [[0, 1], [0]])
    with pytest.raises(ValueError):
        cudf.MultiIndex(levels, [[0, 1], [0]])
    # Didn't pass levels and codes
    with pytest.raises(TypeError):
        pd.MultiIndex()
    with pytest.raises(TypeError):
        cudf.MultiIndex()
    # Didn't pass non zero levels and codes
    with pytest.raises(ValueError):
        pd.MultiIndex([], [])
    with pytest.raises(ValueError):
        cudf.MultiIndex([], [])


def test_multiindex_construction():
    levels = [['a', 'b'], ['c', 'd']]
    codes = [[0, 1], [1, 0]]
    pmi = pd.MultiIndex(levels, codes)
    mi = cudf.MultiIndex(levels, codes)
    assert_eq(pmi, mi)
    pmi = pd.MultiIndex(levels=levels, codes=codes)
    mi = cudf.MultiIndex(levels=levels, codes=codes)
    assert_eq(pmi, mi)


def test_multiindex_types():
    codes = [[0, 1], [1, 0]]
    levels = [[0, 1], [2, 3]]
    pmi = pd.MultiIndex(levels, codes)
    mi = cudf.MultiIndex(levels, codes)
    assert_eq(pmi, mi)
    levels = [[1.2, 2.1], [1.3, 3.1]]
    pmi = pd.MultiIndex(levels, codes)
    mi = cudf.MultiIndex(levels, codes)
    assert_eq(pmi, mi)
    levels = [['a', 'b'], ['c', 'd']]
    pmi = pd.MultiIndex(levels, codes)
    mi = cudf.MultiIndex(levels, codes)
    assert_eq(pmi, mi)


def test_multiindex_df_assignment():
    pdf = pd.DataFrame({'x': [1, 2, 3]})
    gdf = cudf.from_pandas(pdf)
    pdf.index = pd.MultiIndex(levels=[['a', 'b'], ['c', 'd']],
                              codes=[[0, 1, 0], [1, 0, 1]])
    gdf.index = cudf.MultiIndex(levels=[['a', 'b'], ['c', 'd']],
                                codes=[[0, 1, 0], [1, 0, 1]])
    assert_eq(pdf, gdf)


def test_multiindex_series_assignment():
    ps = pd.Series([1, 2, 3])
    gs = cudf.from_pandas(ps)
    ps.index = pd.MultiIndex(levels=[['a', 'b'], ['c', 'd']],
                             codes=[[0, 1, 0], [1, 0, 1]])
    gs.index = cudf.MultiIndex(levels=[['a', 'b'], ['c', 'd']],
                               codes=[[0, 1, 0], [1, 0, 1]])
    assert_eq(ps, gs)


def test_string_index():
    from cudf.dataframe.index import StringIndex, StringColumn
    pdf = pd.DataFrame(np.random.rand(5, 5))
    gdf = cudf.from_pandas(pdf)
    stringIndex = ['a', 'b', 'c', 'd', 'e']
    pdf.index = stringIndex
    gdf.index = stringIndex
    assert_eq(pdf, gdf)
    stringIndex = np.array(['a', 'b', 'c', 'd', 'e'])
    pdf.index = stringIndex
    gdf.index = stringIndex
    assert_eq(pdf, gdf)
    stringIndex = StringIndex(['a', 'b', 'c', 'd', 'e'], name='name')
    pdf.index = stringIndex
    gdf.index = stringIndex
    assert_eq(pdf, gdf)
    stringIndex = StringColumn(['a', 'b', 'c', 'd', 'e'], name='name')
    pdf.index = stringIndex
    gdf.index = stringIndex
    assert_eq(pdf, gdf)


def test_multiindex_loc():
    pdf = pd.DataFrame(np.random.rand(5, 5))
    gdf = cudf.from_pandas(pdf)
    pdfIndex = pd.MultiIndex([['a', 'b', 'c'],
                              ['house', 'store', 'forest'],
                              ['clouds', 'clear', 'storm'],
                              ['fire', 'smoke', 'clear']],
                             [[0, 0, 1, 1, 2],
                              [1, 1, 0, 0, 2],
                              [2, 2, 2, 0, 1],
                              [0, 1, 2, 0, 1]])
    pdfIndex.names = ['alpha', 'location', 'weather', 'sign']
    gdfIndex = cudf.from_pandas(pdfIndex)
    assert_eq(pdfIndex, gdfIndex)
    pdf.index = pdfIndex
    gdf.index = gdfIndex
    assert_eq(pdf.loc[('a', 'store', 'storm')],
              gdf.loc[('a', 'store', 'storm')])
    assert_eq(pdf.loc[('a', 'store')],
              gdf.loc[('a', 'store')])
    assert_eq(pdf.loc[('a')],
              gdf.loc[('a')])
    assert_eq(pdf.loc[('a', 'store', 'storm', 'smoke')],
              gdf.loc[('a', 'store', 'storm', 'smoke')])
    assert_eq(pdf.loc[('b', 'house')],
              gdf.loc[('b', 'house')])
    # assert_eq(pdf.loc[('a', 'store'): ('b', 'house')],
    #           gdf.loc[('a', 'store'): ('b', 'house')])


def test_multiindex_columns():
    pdf = pd.DataFrame(np.random.rand(5, 5))
    gdf = cudf.from_pandas(pdf)
    pdfIndex = pd.MultiIndex([['a', 'b', 'c'],
                              ['house', 'store', 'forest'],
                              ['clouds', 'clear', 'storm']],
                             [[0, 0, 1, 1, 2],
                              [1, 1, 0, 0, 2],
                              [2, 2, 2, 0, 1]])
    pdfIndex.names = ['alpha', 'location', 'weather']
    gdfIndex = cudf.from_pandas(pdfIndex)
    assert_eq(pdfIndex, gdfIndex)
    pdf.columns = pdfIndex
    gdf.columns = gdfIndex
    print(pdf)
    print(gdf.columns)
    print(dir(gdf))
    assert_eq(pdf[('a', 'store')],
              gdf[('a', 'store')])
    assert_eq(pdf[('b', 'house')],
              gdf[('b', 'house')])
    assert_eq(pdf[('a', 'store'), ('b', 'house')],
              gdf[('a', 'store'), ('b', 'house')])
    assert_eq(pdf.loc[slice('a', 'b'), slice('store')])


def test_multiindex_from_tuples():
    arrays = [['a', 'a', 'b', 'b'],
              ['house', 'store', 'house', 'store']]
    tuples = list(zip(*arrays))
    pmi = pd.MultiIndex.from_tuples(tuples, names=['alpha', 'location'])
    gmi = cudf.MultiIndex.from_tuples(tuples, names=['alpha', 'location'])
    assert_eq(pmi, gmi)


def test_multiindex_from_dataframe():
    pdf = pd.DataFrame([['a', 'a', 'b', 'b'],
                        ['house', 'store', 'house', 'store']])
    gdf = cudf.from_pandas(pdf)
    pmi = pd.MultiIndex.from_dataframe(pdf, names=['alpha', 'location'])
    gmi = cudf.MultiIndex.from_dataframe(gdf, names=['alpha', 'location'])
    assert_eq(pmi, gmi)


def test_multiindex_from_product():
    arrays = [['a', 'a', 'b', 'b'],
              ['house', 'store', 'house', 'store']]
    pmi = pd.MultiIndex.from_product(arrays)
    gmi = cudf.MultiIndex.from_product(arrays)
    assert_eq(pmi, gmi)
