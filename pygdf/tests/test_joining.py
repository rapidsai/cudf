from itertools import product

import pytest

import numpy as np

from pygdf.dataframe import DataFrame, Series


# sort_nelem_args = [2, 257]
# sort_dtype_args = [np.int32, np.int64, np.float32, np.float64]

def make_params():
    hows = 'left,inner'.split(',')
    aa = [0, 0, 4, 5, 5]
    bb = [0, 0, 2, 3, 5]
    for how in hows:
        yield (aa, bb, how)

    aa = [0, 0, 1, 2, 3]
    bb = [0, 1, 2, 2, 3]
    for how in hows:
        yield (aa, bb, how)


@pytest.mark.parametrize('aa,bb,how', make_params())
def test_dataframe_join(aa, bb, how):
    df = DataFrame()
    df['a'] = aa
    df['b'] = bb

    def work(df):
        df1 = df.set_index('a')
        df2 = df.set_index('b')
        joined = df1.join(df2, how=how)
        return joined

    expect = work(df.to_pandas())
    got = work(df)

    print(expect)
    print(got.to_string(nrows=None))

    assert list(expect.columns) == list(got.columns)
    assert list(expect.index) == list(got.index)
    assert list(expect['b']) == list(got['b'])
    _check_series(expect['b'], got['b'])
    _check_series(expect['a'], got['a'])


def _check_series(expect, got):
    magic = 0xdeadbeaf
    direct_equal = list(expect) == list(got)
    nanfilled_equal = list(expect.fillna(magic)) == list(got.fillna(magic))
    msg = "direct_equal={}, nanfilled_equal={}".format(direct_equal,
                                                       nanfilled_equal)
    assert direct_equal or nanfilled_equal, msg

