from timeit import default_timer as timer

import pytest

import numpy as np

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

