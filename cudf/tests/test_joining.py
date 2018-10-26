# Copyright (c) 2018, NVIDIA CORPORATION.

from timeit import default_timer as timer

import pytest

import numpy as np
import pandas as pd

from cudf.dataframe import DataFrame


def make_params():
    np.random.seed(0)

    hows = 'left,inner,outer,right'.split(',')
    methods = 'hash,sort'.split(',')

    # Test specific cases (1)
    aa = [0, 0, 4, 5, 5]
    bb = [0, 0, 2, 3, 5]
    for how in hows:
        if how in ['left', 'inner', 'right']:
            for method in methods:
                yield (aa, bb, how, method)
        else:
            yield(aa, bb, how, 'sort')

    # Test specific cases (2)
    aa = [0, 0, 1, 2, 3]
    bb = [0, 1, 2, 2, 3]
    for how in hows:
        if how in ['left', 'inner', 'right']:
            for method in methods:
                yield (aa, bb, how, method)
        else:
            yield(aa, bb, how, 'sort')

    # Test large random integer inputs
    aa = np.random.randint(0, 50, 100)
    bb = np.random.randint(0, 50, 100)
    for how in hows:
        if how in ['left', 'inner', 'right']:
            for method in methods:
                yield (aa, bb, how, method)
        else:
            yield(aa, bb, how, 'sort')

    # Test floating point inputs
    aa = np.random.random(50)
    bb = np.random.random(50)
    for how in hows:
        if how in ['left', 'inner', 'right']:
            for method in methods:
                yield (aa, bb, how, method)
        else:
            yield(aa, bb, how, 'sort')


@pytest.mark.parametrize('aa,bb,how,method', make_params())
def test_dataframe_join_how(aa, bb, how, method):
    df = DataFrame()
    df['a'] = aa
    df['b'] = bb

    def work_pandas(df):
        ts = timer()
        df1 = df.set_index('a')
        df2 = df.set_index('b')
        joined = df1.join(df2, how=how, sort=True)
        te = timer()
        print('timing', type(df), te - ts)
        return joined

    def work_gdf(df):
        ts = timer()
        df1 = df.set_index('a')
        df2 = df.set_index('b')
        joined = df1.join(df2, how=how, sort=True, method=method)
        te = timer()
        print('timing', type(df), te - ts)
        return joined

    expect = work_pandas(df.to_pandas())
    got = work_gdf(df)
    expecto = expect.copy()
    goto = got.copy()

    # Type conversion to handle NoneType
    expectb = expect.b
    expecta = expect.a
    gotb = got.b
    gota = got.a
    got.drop_column('b')
    got.add_column('b', gotb.astype(np.float64).fillna(np.nan))
    got.drop_column('a')
    got.add_column('a', gota.astype(np.float64).fillna(np.nan))
    expect.drop(['b'], axis=1)
    expect['b'] = expectb.astype(np.float64).fillna(np.nan)
    expect.drop(['a'], axis=1)
    expect['a'] = expecta.astype(np.float64).fillna(np.nan)

    # print(expect)
    # print(got.to_string(nrows=None))

    assert list(expect.columns) == list(got.columns)
    # test disabled until libgdf sort join gets updated with new api
    if method == 'hash':
        assert np.all(expect.index.values == got.index.values)
        if(how != 'outer'):
            pd.util.testing.assert_frame_equal(
                got.to_pandas().sort_values(['b', 'a']).reset_index(drop=True),
                expect.sort_values(['b', 'a']).reset_index(drop=True))
        # if(how=='right'):
        #     _sorted_check_series(expect['a'], expect['b'],
        #                          got['a'], got['b'])
        # else:
        #     _sorted_check_series(expect['b'], expect['a'], got['b'],
        #                          got['a'])
        else:
            _check_series(expecto['b'].fillna(-1), goto['b'].fillna(-1))
            _check_series(expecto['a'].fillna(-1), goto['a'].fillna(-1))


def _check_series(expect, got):
    magic = 0xdeadbeaf
    # print("expect\n", expect)
    # print("got\n", got.to_string(nrows=None))
    direct_equal = np.all(expect.values == got.to_array())
    nanfilled_equal = np.all(expect.fillna(magic).values ==
                             got.fillna(magic).to_array())
    msg = "direct_equal={}, nanfilled_equal={}".format(direct_equal,
                                                       nanfilled_equal)
    assert direct_equal or nanfilled_equal, msg


def test_dataframe_join_suffix():
    np.random.seed(0)

    df = DataFrame()
    for k in 'abc':
        df[k] = np.random.randint(0, 5, 5)

    left = df.set_index('a')
    right = df.set_index('c')
    with pytest.raises(ValueError) as raises:
        left.join(right)
    raises.match("there are overlapping columns but lsuffix"
                 " and rsuffix are not defined")

    got = left.join(right, lsuffix='_left', rsuffix='_right', sort=True)
    # Get expected value
    pddf = df.to_pandas()
    expect = pddf.set_index('a').join(pddf.set_index('c'),
                                      lsuffix='_left', rsuffix='_right')
    # Check
    assert list(expect.columns) == list(got.columns)
    assert np.all(expect.index.values == got.index.values)
    for k in expect.columns:
        _check_series(expect[k].fillna(-1), got[k].fillna(-1))


def test_dataframe_join_cats():
    lhs = DataFrame()
    lhs['a'] = pd.Categorical(list('aababcabbc'), categories=list('abc'))
    lhs['b'] = bb = np.arange(len(lhs))
    lhs = lhs.set_index('a')

    rhs = DataFrame()
    rhs['a'] = pd.Categorical(list('abcac'), categories=list('abc'))
    rhs['c'] = cc = np.arange(len(rhs))
    rhs = rhs.set_index('a')

    got = lhs.join(rhs)
    expect = lhs.to_pandas().join(rhs.to_pandas())

    # Note: pandas make a object Index after joining
    pd.util.testing.assert_frame_equal(
        got.sort_values(by='b')
        .to_pandas()
        .sort_index()
        .reset_index(drop=True),
        expect.reset_index(drop=True))

    # Just do some rough checking here.
    assert list(got.columns) == ['b', 'c']
    assert len(got) > 0
    assert set(got.index.values) & set('abc')
    assert set(got['b']) & set(bb)
    assert set(got['c']) & set(cc)


@pytest.mark.parametrize('how', ['left', 'right', 'inner', 'outer'])
def test_dataframe_join_mismatch_cats(how):
    pdf1 = pd.DataFrame({"join_col": ["a", "b", "c", "d", "e"],
                         "data_col_left": [10, 20, 30, 40, 50]})
    pdf2 = pd.DataFrame({"join_col": ["c", "e", "f"],
                         "data_col_right": [6, 7, 8]})

    pdf1["join_col"] = pdf1["join_col"].astype("category")
    pdf2["join_col"] = pdf2["join_col"].astype("category")

    gdf1 = DataFrame.from_pandas(pdf1)
    gdf2 = DataFrame.from_pandas(pdf2)

    gdf1 = gdf1.set_index("join_col")
    gdf2 = gdf2.set_index("join_col")

    pdf1 = pdf1.set_index('join_col')
    pdf2 = pdf2.set_index('join_col')
    join_gdf = gdf1.join(gdf2, how=how, sort=True, method='hash')
    join_pdf = pdf1.join(pdf2, how=how)

    got = join_gdf.to_pandas()
    expect = join_pdf.fillna(-1)  # note: cudf join doesn't mask NA

    # cudf creates the columns in different order than pandas for right join
    if how == 'right':
        got = got[['data_col_left', 'data_col_right']]

    expect.data_col_right = expect.data_col_right.astype(np.int64)
    expect.data_col_left = expect.data_col_left.astype(np.int64)
    pd.util.testing.assert_frame_equal(got, expect, check_names=False,
                                       check_index_type=False,
                                       # For inner joins, pandas returns
                                       # weird categories.
                                       check_categorical=how != 'inner')
    assert list(got.index) == list(expect.index)


def test_dataframe_multi_column_join():
    np.random.seed(0)

    # Make GDF
    df_left = DataFrame()
    nelem = 500
    df_left['key1'] = np.random.randint(0, 30, nelem)
    df_left['key2'] = np.random.randint(0, 50, nelem)
    df_left['val1'] = np.arange(nelem)

    df_right = DataFrame()
    nelem = 500
    df_right['key1'] = np.random.randint(0, 30, nelem)
    df_right['key2'] = np.random.randint(0, 50, nelem)
    df_right['val1'] = np.arange(nelem)

    # Make pandas DF
    pddf_left = df_left.to_pandas()
    pddf_right = df_right.to_pandas()

    # Expected result
    pddf_joined = pddf_left.merge(pddf_right, on=['key1', 'key2'], how='left',
                                  sort=True)

    # Test (doesn't check for ordering)
    join_result = df_left.merge(df_right, on=['key1', 'key2'], how='left')

    for col in list(pddf_joined.columns):
        if(col.count('_y') > 0):
            join_result[col] = (join_result[col]
                                .astype(np.float64)
                                .fillna(np.nan))

    pd.util.testing.assert_frame_equal(
        join_result
        .to_pandas()
        .sort_values(list(pddf_joined.columns))
        .reset_index(drop=True),
        pddf_joined)
