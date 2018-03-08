from timeit import default_timer as timer

import pytest

import numpy as np
import pandas as pd

from pygdf.dataframe import DataFrame


def make_params():
    np.random.seed(0)

    hows = 'left,inner,outer,right'.split(',')

    # Test specific cases (1)
    aa = [0, 0, 4, 5, 5]
    bb = [0, 0, 2, 3, 5]
    for how in hows:
        yield (aa, bb, how)

    # Test specific cases (2)
    aa = [0, 0, 1, 2, 3]
    bb = [0, 1, 2, 2, 3]
    for how in hows:
        yield (aa, bb, how)

    # Test large random integer inputs
    aa = np.random.randint(0, 50, 100)
    bb = np.random.randint(0, 50, 100)
    for how in hows:
        yield (aa, bb, how)

    # Test floating point inputs
    aa = np.random.random(50)
    bb = np.random.random(50)
    for how in hows:
        yield (aa, bb, how)


@pytest.mark.parametrize('aa,bb,how', make_params())
def test_dataframe_join_how(aa, bb, how):
    df = DataFrame()
    df['a'] = aa
    df['b'] = bb

    def work(df):
        ts = timer()
        df1 = df.set_index('a')
        df2 = df.set_index('b')
        joined = df1.join(df2, how=how, sort=True)
        te = timer()
        print('timing', type(df), te - ts)
        return joined

    expect = work(df.to_pandas())
    got = work(df)

    # print(expect)
    # print(got.to_string(nrows=None))

    assert list(expect.columns) == list(got.columns)
    assert np.all(expect.index.values == got.index.values)
    _check_series(expect['b'], got['b'])
    _check_series(expect['a'], got['a'])


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

    got = left.join(right, lsuffix='_left', rsuffix='_right')
    # Get expected value
    pddf = df.to_pandas()
    expect = pddf.set_index('a').join(pddf.set_index('c'),
                                      lsuffix='_left', rsuffix='_right')
    # Check
    assert list(expect.columns) == list(got.columns)
    assert np.all(expect.index.values == got.index.values)
    for k in expect.columns:
        _check_series(expect[k], got[k])


def test_dataframe_join_cats():
    ldf = DataFrame()
    ldf['a'] = pd.Categorical(list('aababcabbc'), categories=list('abc'))
    ldf['b'] = bb = np.arange(len(ldf))
    lhs = ldf.set_index('a')

    rdf = DataFrame()
    rdf['a'] = pd.Categorical(list('abcac'), categories=list('abc'))
    rdf['c'] = cc = np.arange(len(rdf))
    rhs = rdf.set_index('a')

    got = lhs.join(rhs)
    expect = lhs.to_pandas().join(rhs.to_pandas())

    # Note: pandas make a object Index after joining
    pd.util.testing.assert_frame_equal(got.to_pandas().reset_index(drop=True),
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
    join_gdf = gdf1.join(gdf2, how=how)
    join_pdf = pdf1.join(pdf2, how=how)

    got = join_gdf.to_pandas()
    expect = join_pdf.fillna(-1)  # note: pygdf join doesn't mask NA

    expect.data_col_right = expect.data_col_right.astype(np.int64)
    expect.data_col_left = expect.data_col_left.astype(np.int64)
    pd.util.testing.assert_frame_equal(got, expect, check_names=False,
                                       check_index_type=False,
                                       # For inner joins, pandas return weird categories.
                                       check_categorical=how != 'inner')
    assert list(got.index) == list(expect.index)
