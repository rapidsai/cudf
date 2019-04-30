# Copyright (c) 2018, NVIDIA CORPORATION.

import pytest

import cudf
import numpy as np
import pandas as pd

from cudf.dataframe import DataFrame
from cudf.dataframe import Series
from cudf.tests.utils import assert_eq


def make_frame(dataframe_class, nelem, seed=0, extra_levels=(), extra_vals=()):
    np.random.seed(seed)

    df = dataframe_class()

    df['x'] = np.random.randint(0, 5, nelem)
    df['y'] = np.random.randint(0, 3, nelem)
    for lvl in extra_levels:
        df[lvl] = np.random.randint(0, 2, nelem)

    df['val'] = np.random.random(nelem)
    for val in extra_vals:
        df[val] = np.random.random(nelem)

    return df


def get_methods():
    for method in ['cudf', 'hash']:
        yield method


def get_nelem():
    for elem in [2, 3, 1000]:
        yield elem


@pytest.fixture
def gdf():
    return DataFrame({'x': [1, 2, 3], 'y': [0, 1, 1]})


@pytest.fixture
def pdf(gdf):
    return gdf.to_pandas()


@pytest.mark.parametrize('as_index', [True, False])
def test_groupby_as_index_single_agg(pdf, gdf, as_index):
    gdf = gdf.groupby('y', as_index=as_index).agg({'x': 'mean'})
    pdf = pdf.groupby('y', as_index=as_index).agg({'x': 'mean'})
    assert_eq(pdf, gdf)


def test_groupby_default(pdf, gdf):
    gdf = gdf.groupby('y').agg({'x': 'mean'})
    pdf = pdf.groupby('y').agg({'x': 'mean'})
    assert_eq(pdf, gdf)


def test_groupby_getitem_styles():
    pdf = pd.DataFrame({'x': [1, 3, 1], 'y': [1, 2, 3]})
    gdf = cudf.from_pandas(pdf)
    assert_eq(gdf.groupby('x')['y'].sum(),
              pdf.groupby('x')['y'].sum())
    assert_eq(pdf.groupby('x').y.sum(),
              gdf.groupby('x').y.sum())
    assert_eq(pdf.groupby('x')[['y']].sum(),
              gdf.groupby('x')[['y']].sum())


@pytest.mark.parametrize('nelem', get_nelem())
@pytest.mark.parametrize('method', get_methods())
def test_groupby_mean(nelem, method):
    got_df = make_frame(DataFrame, nelem=nelem).groupby(
        ['x', 'y'], method=method).mean()
    expect_df = make_frame(pd.DataFrame,
                           nelem=nelem).groupby(['x', 'y']).mean()
    if method == "cudf":
        got = np.sort(got_df['val'].to_array())
        expect = np.sort(expect_df['val'].values)
        np.testing.assert_array_almost_equal(expect, got)
    else:
        assert_eq(got_df, expect_df)


@pytest.mark.parametrize('nelem', get_nelem())
@pytest.mark.parametrize('method', get_methods())
def test_groupby_mean_3level(nelem, method):
    lvls = 'z'
    bys = list('xyz')
    got_df = make_frame(DataFrame, nelem=nelem,
                        extra_levels=lvls).groupby(bys, method=method).mean()
    expect_df = make_frame(pd.DataFrame, nelem=nelem,
                           extra_levels=lvls).groupby(bys).mean()
    if method == "cudf":
        got = np.sort(got_df['val'].to_array())
        expect = np.sort(expect_df['val'].values)
        np.testing.assert_array_almost_equal(expect, got)
    else:
        assert_eq(got_df, expect_df)


@pytest.mark.parametrize('nelem', get_nelem())
@pytest.mark.parametrize('method', get_methods())
def test_groupby_agg_mean_min(nelem, method):
    got_df = make_frame(DataFrame, nelem=nelem).groupby(
        ['x', 'y'], method=method).agg(['mean', 'min'])
    expect_df = make_frame(pd.DataFrame, nelem=nelem).groupby(
        ['x', 'y']).agg(['mean', 'min'])
    if method == "cudf":
        got_mean = np.sort(got_df['val_mean'].to_array())
        got_min = np.sort(got_df['val_min'].to_array())
        expect_mean = np.sort(expect_df['val', 'mean'].values)
        expect_min = np.sort(expect_df['val', 'min'].values)
        # verify
        np.testing.assert_array_almost_equal(expect_mean, got_mean)
        np.testing.assert_array_almost_equal(expect_min, got_min)
    else:
        assert_eq(expect_df, got_df)


@pytest.mark.parametrize('nelem', get_nelem())
@pytest.mark.parametrize('method', get_methods())
def test_groupby_agg_min_max_dictargs(nelem, method):
    # gdf (Note: lack of multindex)
    got_df = make_frame(DataFrame, nelem=nelem, extra_vals='ab').groupby(
        ['x', 'y'], method=method).agg({'a': 'min', 'b': 'max'})
    expect_df = make_frame(pd.DataFrame, nelem=nelem, extra_vals='ab').groupby(
        ['x', 'y']).agg({'a': 'min', 'b': 'max'})
    if method == "cudf":
        got_min = np.sort(got_df['a'].to_array())
        got_max = np.sort(got_df['b'].to_array())
        expect_min = np.sort(expect_df['a'].values)
        expect_max = np.sort(expect_df['b'].values)
        # verify
        np.testing.assert_array_almost_equal(expect_min, got_min)
        np.testing.assert_array_almost_equal(expect_max, got_max)
    else:
        assert_eq(expect_df, got_df)


@pytest.mark.parametrize('method', get_methods())
def test_groupby_cats(method):
    df = DataFrame()
    df['cats'] = pd.Categorical(list('aabaacaab'))
    df['vals'] = np.random.random(len(df))

    cats = np.asarray(list(df['cats']))
    vals = df['vals'].to_array()

    grouped = df.groupby(['cats'], method=method, as_index=False).mean()

    got_vals = grouped['vals']

    got_cats = grouped['cats']

    for c, v in zip(got_cats, got_vals):
        print(c, v)
        expect = vals[cats == c].mean()
        np.testing.assert_almost_equal(v, expect)


def test_groupby_iterate_groups():
    np.random.seed(0)
    df = DataFrame()
    nelem = 20
    df['key1'] = np.random.randint(0, 3, nelem)
    df['key2'] = np.random.randint(0, 2, nelem)
    df['val1'] = np.random.random(nelem)
    df['val2'] = np.random.random(nelem)

    def assert_values_equal(arr):
        np.testing.assert_array_equal(arr[0], arr)

    for grp in df.groupby(['key1', 'key2'], method="cudf"):
        pddf = grp.to_pandas()
        for k in 'key1,key2'.split(','):
            assert_values_equal(pddf[k].values)


def test_groupby_as_df():
    np.random.seed(0)
    df = DataFrame()
    nelem = 20
    df['key1'] = np.random.randint(0, 3, nelem)
    df['key2'] = np.random.randint(0, 2, nelem)
    df['val1'] = np.random.random(nelem)
    df['val2'] = np.random.random(nelem)

    def assert_values_equal(arr):
        np.testing.assert_array_equal(arr[0], arr)

    df, segs = df.groupby(['key1', 'key2'], method="cudf").as_df()
    for s, e in zip(segs, list(segs[1:]) + [None]):
        grp = df[s:e]
        pddf = grp.to_pandas()
        for k in 'key1,key2'.split(','):
            assert_values_equal(pddf[k].values)


def test_groupby_apply():
    np.random.seed(0)
    df = DataFrame()
    nelem = 20
    df['key1'] = np.random.randint(0, 3, nelem)
    df['key2'] = np.random.randint(0, 2, nelem)
    df['val1'] = np.random.random(nelem)
    df['val2'] = np.random.random(nelem)

    expect_grpby = df.to_pandas().groupby(['key1', 'key2'],
                                          as_index=False)
    got_grpby = df.groupby(['key1', 'key2'], method="cudf")

    def foo(df):
        df['out'] = df['val1'] + df['val2']
        return df

    expect = expect_grpby.apply(foo)
    expect = expect.sort_values(['key1', 'key2']).reset_index(drop=True)

    got = got_grpby.apply(foo).to_pandas()
    pd.util.testing.assert_frame_equal(expect, got)


def test_groupby_apply_grouped():
    from numba import cuda

    np.random.seed(0)
    df = DataFrame()
    nelem = 20
    df['key1'] = np.random.randint(0, 3, nelem)
    df['key2'] = np.random.randint(0, 2, nelem)
    df['val1'] = np.random.random(nelem)
    df['val2'] = np.random.random(nelem)

    expect_grpby = df.to_pandas().groupby(['key1', 'key2'], as_index=False)
    got_grpby = df.groupby(['key1', 'key2'], method="cudf")

    def foo(key1, val1, com1, com2):
        for i in range(cuda.threadIdx.x, len(key1), cuda.blockDim.x):
            com1[i] = key1[i] * 10000 + val1[i]
            com2[i] = i

    got = got_grpby.apply_grouped(foo,
                                  incols=['key1', 'val1'],
                                  outcols={'com1': np.float64,
                                           'com2': np.int32},
                                  tpb=8)

    got = got.to_pandas()

    # Get expected result by emulating the operation in pandas
    def emulate(df):
        df['com1'] = df.key1 * 10000 + df.val1
        df['com2'] = np.arange(len(df), dtype=np.int32)
        return df
    expect = expect_grpby.apply(emulate)
    expect = expect.sort_values(['key1', 'key2']).reset_index(drop=True)

    pd.util.testing.assert_frame_equal(expect, got)


@pytest.mark.parametrize('nelem', [100, 500])
@pytest.mark.parametrize('func', ['mean', 'std', 'var', 'min',
                                  'max', 'count', 'sum'])
@pytest.mark.parametrize('method', get_methods())
def test_groupby_cudf_2keys_agg(nelem, func, method):
    # skip unimplemented aggs:
    if func in ['var', 'std']:
        if method in ['hash', 'sort']:
            pytest.skip()

    got_df = make_frame(DataFrame, nelem=nelem)\
        .groupby(['x', 'y'], method=method).agg(func)

    got_agg = np.sort(got_df['val'].to_array())
    # pandas
    expect_df = make_frame(pd.DataFrame, nelem=nelem)\
        .groupby(['x', 'y']).agg(func)
    if method == 'cudf':
        expect_agg = np.sort(expect_df['val'].values)
        # verify
        np.testing.assert_array_almost_equal(expect_agg, got_agg)
    else:
        assert_eq(got_df, expect_df)


@pytest.mark.parametrize('agg', ['min', 'max', 'count', 'sum', 'mean'])
def test_series_groupby(agg):
    s = pd.Series([1, 2, 3])
    g = Series([1, 2, 3])
    sg = s.groupby(s // 2)
    gg = g.groupby(g // 2)
    sa = getattr(sg, agg)()
    ga = getattr(gg, agg)()
    assert_eq(sa, ga)


@pytest.mark.xfail(reason="Prefixed column names are not removed yet")
@pytest.mark.parametrize('agg', ['min', 'max', 'count', 'sum', 'mean'])
def test_series_groupby_agg(agg):
    s = pd.Series([1, 2, 3])
    g = Series([1, 2, 3])
    sg = s.groupby(s // 2).agg(agg)
    gg = g.groupby(g // 2).agg(agg)
    assert_eq(sg, gg)


@pytest.mark.parametrize('agg', ['min', 'max', 'count', 'sum', 'mean'])
def test_groupby_level_zero(agg):
    pdf = pd.DataFrame({'x': [1, 2, 3]}, index=[0, 1, 1])
    gdf = DataFrame.from_pandas(pdf)
    pdg = pdf.groupby(level=0)
    gdg = gdf.groupby(level=0)
    pdresult = getattr(pdg, agg)()
    gdresult = getattr(gdg, agg)()
    assert_eq(pdresult, gdresult)


@pytest.mark.parametrize('agg', ['min', 'max', 'count', 'sum', 'mean'])
def test_groupby_series_level_zero(agg):
    pdf = pd.Series([1, 2, 3], index=[0, 1, 1])
    gdf = Series.from_pandas(pdf)
    pdg = pdf.groupby(level=0)
    gdg = gdf.groupby(level=0)
    pdresult = getattr(pdg, agg)()
    gdresult = getattr(gdg, agg)()
    assert_eq(pdresult, gdresult)


def test_groupby_column_name():
    pdf = pd.DataFrame({'xx': [1., 2., 3.], 'yy': [1, 2, 3]})
    gdf = DataFrame.from_pandas(pdf)
    g = gdf.groupby('yy')
    p = pdf.groupby('yy')
    gxx = g['xx'].sum()
    pxx = p['xx'].sum()
    assert_eq(pxx, gxx)


def test_groupby_column_numeral():
    pdf = pd.DataFrame({0: [1., 2., 3.], 1: [1, 2, 3]})
    gdf = DataFrame.from_pandas(pdf)
    p = pdf.groupby(1)
    g = gdf.groupby(1)
    pxx = p[0].sum()
    gxx = g[0].sum()
    assert_eq(pxx, gxx)


@pytest.mark.parametrize('series', [[0, 1, 0], [1, 1, 1], [0, 1, 1], [1, 2, 3], [4, 3, 2], [0, 2, 0]])  # noqa: E501
def test_groupby_external_series(series):
    pdf = pd.DataFrame({'x': [1., 2., 3.], 'y': [1, 2, 1]})
    gdf = DataFrame.from_pandas(pdf)
    pxx = pdf.groupby(pd.Series(series)).x.sum()
    gxx = gdf.groupby(cudf.Series(series)).x.sum()
    assert_eq(pxx, gxx)


@pytest.mark.xfail(raises=NotImplementedError,
                   reason="CUDF doesn't support arbitrary series index lengths"
                          "for groupby")
@pytest.mark.parametrize('series', [[0, 1], [1, 1, 1, 1]])
def test_groupby_external_series_incorrect_length(series):
    pdf = pd.DataFrame({'x': [1., 2., 3.], 'y': [1, 2, 1]})
    gdf = DataFrame.from_pandas(pdf)
    pxx = pdf.groupby(pd.Series(series)).x.sum()
    gxx = gdf.groupby(cudf.Series(series)).x.sum()
    assert_eq(pxx, gxx)
