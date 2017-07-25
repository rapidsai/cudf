from itertools import product
from timeit import default_timer as timer

import pytest

import numpy as np

from pygdf.dataframe import DataFrame, Series


# sort_nelem_args = [2, 257]
# sort_dtype_args = [np.int32, np.int64, np.float32, np.float64]

def make_params():
    np.random.seed(0)

    hows = 'left,inner'.split(',')
    aa = [0, 0, 4, 5, 5]
    bb = [0, 0, 2, 3, 5]
    for how in hows:
        yield (aa, bb, how)

    aa = [0, 0, 1, 2, 3]
    bb = [0, 1, 2, 2, 3]
    for how in hows:
        yield (aa, bb, how)

    aa = np.random.randint(0, 50, 100)
    bb = np.random.randint(0, 50, 100)
    for how in hows:
        yield (aa, bb, how)


@pytest.mark.parametrize('aa,bb,how', make_params())
def test_dataframe_join(aa, bb, how):
    df = DataFrame()
    df['a'] = aa
    df['b'] = bb

    def work(df):
        ts = timer()
        df1 = df.set_index('a')
        df2 = df.set_index('b')
        joined = df1.join(df2, how=how)
        te = timer()
        print(type(df), te - ts)
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
    direct_equal = np.all(expect.values == got.to_array())
    nanfilled_equal = np.all(expect.fillna(magic).values ==
                             got.fillna(magic).to_array())
    msg = "direct_equal={}, nanfilled_equal={}".format(direct_equal,
                                                       nanfilled_equal)
    assert direct_equal or nanfilled_equal, msg

