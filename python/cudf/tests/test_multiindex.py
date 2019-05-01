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
    pmi = pd.MultiIndex(levels, codes)
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
    pdf.index = pd.MultiIndex([['a', 'b'], ['c', 'd']],
                              [[0, 1, 0], [1, 0, 1]])
    gdf.index = cudf.MultiIndex(levels=[['a', 'b'], ['c', 'd']],
                                codes=[[0, 1, 0], [1, 0, 1]])
    assert_eq(pdf, gdf)


def test_multiindex_series_assignment():
    ps = pd.Series([1, 2, 3])
    gs = cudf.from_pandas(ps)
    ps.index = pd.MultiIndex([['a', 'b'], ['c', 'd']],
                             [[0, 1, 0], [1, 0, 1]])
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


def test_multiindex_row_shape():
    pdf = pd.DataFrame(np.random.rand(0, 5))
    gdf = cudf.from_pandas(pdf)
    pdfIndex = pd.MultiIndex([['a', 'b', 'c']],
                             [[0]])
    pdfIndex.names = ['alpha']
    gdfIndex = cudf.from_pandas(pdfIndex)
    assert_eq(pdfIndex, gdfIndex)
    with pytest.raises(ValueError):
        pdf.index = pdfIndex
    with pytest.raises(ValueError):
        gdf.index = gdfIndex


@pytest.fixture
def pdf():
    return pd.DataFrame(np.random.rand(7, 5))


@pytest.fixture
def gdf(pdf):
    return cudf.from_pandas(pdf)


@pytest.fixture
def pdfIndex():
    pdfIndex = pd.MultiIndex([['a', 'b', 'c'],
                              ['house', 'store', 'forest'],
                              ['clouds', 'clear', 'storm'],
                              ['fire', 'smoke', 'clear']],
                             [[0, 0, 0, 0, 1, 1, 2],
                              [1, 1, 1, 1, 0, 0, 2],
                              [0, 0, 2, 2, 2, 0, 1],
                              [0, 0, 0, 1, 2, 0, 1]])
    pdfIndex.names = ['alpha', 'location', 'weather', 'sign']
    return pdfIndex


def test_from_pandas(pdf, pdfIndex):
    pdf.index = pdfIndex
    gdf = cudf.from_pandas(pdf)
    assert_eq(pdf, gdf)


def test_series_multiindex(pdfIndex):
    ps = pd.Series(np.random.rand(7))
    gs = cudf.from_pandas(ps)
    ps.index = pdfIndex
    gs.index = cudf.from_pandas(pdfIndex)
    assert_eq(ps, gs)


def test_multiindex_take(pdf, gdf, pdfIndex):
    gdfIndex = cudf.from_pandas(pdfIndex)
    pdf.index = pdfIndex
    gdf.index = gdfIndex
    assert_eq(pdf.index.take([0]), gdf.index.take([0]))
    assert_eq(pdf.index.take(np.array([0])), gdf.index.take(np.array([0])))
    from cudf import Series
    assert_eq(pdf.index.take(Series([0])), gdf.index.take(Series([0])))
    assert_eq(pdf.index.take([0, 1]), gdf.index.take([0, 1]))
    assert_eq(pdf.index.take(np.array([0, 1])),
              gdf.index.take(np.array([0, 1])))
    assert_eq(pdf.index.take(Series([0, 1])), gdf.index.take(Series([0, 1])))


def test_multiindex_getitem(pdf, gdf, pdfIndex):
    gdfIndex = cudf.from_pandas(pdfIndex)
    pdf.index = pdfIndex
    gdf.index = gdfIndex
    assert_eq(pdf.index[0], gdf.index[0])


def test_multiindex_loc(pdf, gdf, pdfIndex):
    gdfIndex = cudf.from_pandas(pdfIndex)
    assert_eq(pdfIndex, gdfIndex)
    pdf.index = pdfIndex
    gdf.index = gdfIndex
    # return 2 rows, 0 remaining keys = dataframe with entire index
    assert_eq(pdf.loc[('a', 'store', 'clouds', 'fire')],
              gdf.loc[('a', 'store', 'clouds', 'fire')])
    # return 2 rows, 1 remaining key = dataframe with n-k index columns
    assert_eq(pdf.loc[('a', 'store', 'storm')],
              gdf.loc[('a', 'store', 'storm')])
    # return 2 rows, 2 remaining keys = dataframe with n-k index columns
    assert_eq(pdf.loc[('a', 'store')],
              gdf.loc[('a', 'store')])
    assert_eq(pdf.loc[('b', 'house')],
              gdf.loc[('b', 'house')])
    # return 2 rows, n-1 remaining keys = dataframe with n-k index columns
    assert_eq(pdf.loc[('a',)],
              gdf.loc[('a',)])
    # return 1 row, 0 remaining keys = dataframe with entire index
    assert_eq(pdf.loc[('a', 'store', 'storm', 'smoke')],
              gdf.loc[('a', 'store', 'storm', 'smoke')])
    # return 1 row and 1 remaining key = series
    assert_eq(pdf.loc[('c', 'forest', 'clear')],
              gdf.loc[('c', 'forest', 'clear')])


@pytest.mark.xfail(reason="Slicing MultiIndexes not supported yet",
                   raises=AttributeError)
def test_multiindex_loc_slice(pdf, gdf, pdfIndex):
    gdf = cudf.from_pandas(pdf)
    gdfIndex = cudf.from_pandas(pdfIndex)
    pdf.index = pdfIndex
    gdf.index = gdfIndex
    assert_eq(pdf.loc[('a', 'store'): ('b', 'house')],
              gdf.loc[('a', 'store'): ('b', 'house')])


def test_multiindex_loc_then_column(pdf, gdf, pdfIndex):
    gdfIndex = cudf.from_pandas(pdfIndex)
    assert_eq(pdfIndex, gdfIndex)
    pdf.index = pdfIndex
    gdf.index = gdfIndex
    assert_eq(pdf.loc[('a', 'store', 'clouds', 'fire')][0],
              gdf.loc[('a', 'store', 'clouds', 'fire')][0])


def test_multiindex_loc_rows_0(pdf, gdf, pdfIndex):
    gdfIndex = cudf.from_pandas(pdfIndex)
    pdf.index = pdfIndex
    gdf.index = gdfIndex
    with pytest.raises(KeyError):
        print(pdf.loc[('d',)])
    with pytest.raises(KeyError):
        print(gdf.loc[('d',)])
    assert_eq(pdf, gdf)


def test_multiindex_loc_rows_1_2_key(pdf, gdf, pdfIndex):
    gdfIndex = cudf.from_pandas(pdfIndex)
    pdf.index = pdfIndex
    gdf.index = gdfIndex
    print(pdf.loc[('c', 'forest')])
    print(gdf.loc[('c', 'forest')].to_pandas())
    assert_eq(pdf.loc[('c', 'forest')], gdf.loc[('c', 'forest')])


def test_multiindex_loc_rows_1_1_key(pdf, gdf, pdfIndex):
    gdfIndex = cudf.from_pandas(pdfIndex)
    pdf.index = pdfIndex
    gdf.index = gdfIndex
    print(pdf.loc[('c',)])
    print(gdf.loc[('c',)].to_pandas())
    assert_eq(pdf.loc[('c',)], gdf.loc[('c',)])


def test_multiindex_column_shape():
    pdf = pd.DataFrame(np.random.rand(5, 0))
    gdf = cudf.from_pandas(pdf)
    pdfIndex = pd.MultiIndex([['a', 'b', 'c']],
                             [[0]])
    pdfIndex.names = ['alpha']
    gdfIndex = cudf.from_pandas(pdfIndex)
    assert_eq(pdfIndex, gdfIndex)
    with pytest.raises(ValueError):
        pdf.columns = pdfIndex
    with pytest.raises(ValueError):
        gdf.columns = gdfIndex


def test_multiindex_columns(pdf, gdf, pdfIndex):
    pdf = pdf.T
    gdf = cudf.from_pandas(pdf)
    gdfIndex = cudf.from_pandas(pdfIndex)
    assert_eq(pdfIndex, gdfIndex)
    pdf.columns = pdfIndex
    gdf.columns = gdfIndex
    assert_eq(pdf[('a', 'store', 'clouds', 'fire')],
              gdf[('a', 'store', 'clouds', 'fire')])
    assert_eq(pdf[('a', 'store', 'storm', 'smoke')],
              gdf[('a', 'store', 'storm', 'smoke')])
    assert_eq(pdf[('a', 'store')],
              gdf[('a', 'store')])
    assert_eq(pdf[('b', 'house')],
              gdf[('b', 'house')])
    assert_eq(pdf[('a', 'store', 'storm')],
              gdf[('a', 'store', 'storm')])
    assert_eq(pdf[('a',)],
              gdf[('a',)])
    assert_eq(pdf[('c', 'forest', 'clear')],
              gdf[('c', 'forest', 'clear')])

@pytest.mark.xfail(reason="Slicing MultiIndexes not supported yet",
                   raises=TypeError)
def test_multiindex_column_slice(pdf, gdf, pdfIndex):
    pdf = pdf.T
    gdf = cudf.from_pandas(pdf)
    gdfIndex = cudf.from_pandas(pdfIndex)
    pdf.columns = pdfIndex
    gdf.columns = gdfIndex
    assert_eq(pdf[('a', 'store'): ('b', 'house')],
              gdf[('a', 'store'): ('b', 'house')])


def test_multiindex_from_tuples():
    arrays = [['a', 'a', 'b', 'b'],
              ['house', 'store', 'house', 'store']]
    tuples = list(zip(*arrays))
    pmi = pd.MultiIndex.from_tuples(tuples)
    gmi = cudf.MultiIndex.from_tuples(tuples)
    assert_eq(pmi, gmi)


def test_multiindex_from_dataframe():
    if not hasattr(pd.MultiIndex([[]], [[]]), 'codes'):
        pytest.skip()
    pdf = pd.DataFrame([['a', 'house'], ['a', 'store'],
                        ['b', 'house'], ['b', 'store']])
    gdf = cudf.from_pandas(pdf)
    pmi = pd.MultiIndex.from_frame(pdf, names=['alpha', 'location'])
    gmi = cudf.MultiIndex.from_frame(gdf, names=['alpha', 'location'])
    assert_eq(pmi, gmi)


def test_multiindex_from_product():
    arrays = [['a', 'a', 'b', 'b'],
              ['house', 'store', 'house', 'store']]
    pmi = pd.MultiIndex.from_product(arrays, names=['alpha', 'location'])
    gmi = cudf.MultiIndex.from_product(arrays, names=['alpha', 'location'])
    assert_eq(pmi, gmi)


def test_multiindex_index_and_columns():
    gdf = cudf.DataFrame()
    gdf['x'] = np.random.randint(0, 5, 5)
    gdf['y'] = np.random.randint(0, 5, 5)
    pdf = gdf.to_pandas()
    mi = cudf.MultiIndex(levels=[[0, 1, 2], [3, 4]], codes=[[0, 0, 1, 1, 2],
                         [0, 1, 0, 1, 1]], names=['x', 'y'])
    gdf.index = mi
    mc = cudf.MultiIndex(levels=[['val'], ['mean', 'min']],
                         codes=[[0, 0], [0, 1]])
    gdf.columns = mc
    pdf.index = mi
    pdf.index.names = ['x', 'y']
    pdf.columns = mc
    assert_eq(pdf, gdf)


def test_multiindex_multiple_groupby():
    pdf = pd.DataFrame(
        {
            "a": [4, 17, 4, 9, 5],
            "b": [1, 4, 4, 3, 2],
            "x": np.random.normal(size=5),
        }
    )
    gdf = cudf.DataFrame.from_pandas(pdf)
    pdg = pdf.groupby(['a', 'b']).sum()
    gdg = gdf.groupby(['a', 'b']).sum()
    assert_eq(pdg, gdg)
    pdg = pdf.groupby(['a', 'b']).x.sum()
    gdg = gdf.groupby(['a', 'b']).x.sum()
    assert_eq(pdg, gdg)


@pytest.mark.parametrize(
    "func",
    [
        lambda df: df.groupby(["x", "y"]).z.sum(),
        lambda df: df.groupby(["x", "y"]).sum(),
    ],
)
def test_multi_column(func):
    pdf = pd.DataFrame(
        {
            "x": np.random.randint(0, 5, size=1000),
            "y": np.random.randint(0, 10, size=1000),
            "z": np.random.normal(size=1000),
        }
    )
    gdf = cudf.DataFrame.from_pandas(pdf)

    a = func(pdf)
    b = func(gdf)

    assert_eq(a, b)
