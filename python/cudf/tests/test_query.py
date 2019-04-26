# Copyright (c) 2018, NVIDIA CORPORATION.

from __future__ import print_function, division

import inspect
import datetime
import pytest
import numpy as np
import pandas as pd
from pandas.util.testing import assert_frame_equal
from itertools import product

import cudf
from cudf.utils import queryutils
from cudf.dataframe import DataFrame


_params_query_parser = []
_params_query_parser.append(('a > @b', ('a', '__CUDF_ENVREF__b')))
_params_query_parser.append(('(a + b) <= @c', ('a', 'b', '__CUDF_ENVREF__c')))
_params_query_parser.append(('a > b if a > 0 else b > a', ('a', 'b')))


@pytest.mark.parametrize('text,expect_args', _params_query_parser)
def test_query_parser(text, expect_args):
    info = queryutils.query_parser(text)
    fn = queryutils.query_builder(info, 'myfoo')
    assert callable(fn)
    argspec = inspect.getfullargspec(fn)
    assert tuple(argspec.args) == tuple(expect_args)


params_query_data = list(product([1, 2, 7, 8, 9, 16, 100, 129], range(2)))
params_query_fn = [
    (lambda a, b: a < b, 'a < b'),
    (lambda a, b: a * 2 >= b, 'a * 2 >= b'),
    (lambda a, b: 2 * (a + b) > (a + b) / 2, '2 * (a + b) > (a + b) / 2'),
]


@pytest.mark.parametrize('data,fn',
                         product(params_query_data, params_query_fn))
def test_query(data, fn):
    # prepare
    nelem, seed = data
    expect_fn, query_expr = fn
    np.random.seed(seed)
    df = DataFrame()
    df['a'] = aa = np.arange(nelem)
    df['b'] = bb = np.random.random(nelem) * nelem
    # udt
    expect_mask = expect_fn(aa, bb)
    df2 = df.query(query_expr)
    # check
    assert len(df2) == np.count_nonzero(expect_mask)
    np.testing.assert_array_almost_equal(df2['a'].to_array(), aa[expect_mask])
    np.testing.assert_array_almost_equal(df2['b'].to_array(), bb[expect_mask])


params_query_env_fn = [
    (lambda a, b, c, d: a * c > b + d,
     'a * @c > b + @d'),
    (lambda a, b, c, d: ((a / c) < d) | ((b ** c) > d),
     '((a / @c) < @d) | ((b ** @c) > @d)')
]


@pytest.mark.parametrize('data,fn',
                         product(params_query_data, params_query_env_fn))
def test_query_ref_env(data, fn):
    # prepare
    nelem, seed = data
    expect_fn, query_expr = fn
    np.random.seed(seed)
    df = DataFrame()
    df['a'] = aa = np.arange(nelem)
    df['b'] = bb = np.random.random(nelem) * nelem
    c = 2.3
    d = 1.2
    # udt
    expect_mask = expect_fn(aa, bb, c, d)
    print(expect_mask)
    df2 = df.query(query_expr)
    # check
    assert len(df2) == np.count_nonzero(expect_mask)
    np.testing.assert_array_almost_equal(df2['a'].to_array(), aa[expect_mask])
    np.testing.assert_array_almost_equal(df2['b'].to_array(), bb[expect_mask])


def test_query_env_changing():
    df = DataFrame()
    df['a'] = aa = np.arange(100)
    expr = 'a < @c'
    # first attempt
    c = 10
    got = df.query(expr)
    np.testing.assert_array_equal(aa[aa < c], got['a'].to_array())
    # change env
    c = 50
    got = df.query(expr)
    np.testing.assert_array_equal(aa[aa < c], got['a'].to_array())


def test_query_local_dict():
    df = DataFrame()
    df['a'] = aa = np.arange(100)
    expr = "a < @val"

    got = df.query(expr, local_dict={'val': 10})
    np.testing.assert_array_equal(aa[aa < 10], got['a'].to_array())

    # test for datetime
    df = DataFrame()
    data = np.array(['2018-10-07', '2018-10-08'], dtype='datetime64')
    df['datetimes'] = data
    search_date = datetime.datetime.strptime('2018-10-08', '%Y-%m-%d')
    expr = 'datetimes==@search_date'

    got = df.query(expr, local_dict={'search_date': search_date})
    np.testing.assert_array_equal(data[1], got['datetimes'].to_array())


def test_query_splitted_combine():
    np.random.seed(0)
    df = pd.DataFrame({'x': np.random.randint(0, 5, size=10),
                       'y': np.random.normal(size=10)})
    gdf = DataFrame.from_pandas(df)

    # Split the GDF
    s1 = gdf[:5]
    s2 = gdf[5:]

    # Do the query
    expr = 'x > 2'
    q1 = s1.query(expr)
    q2 = s2.query(expr)
    # Combine
    got = cudf.concat([q1, q2]).to_pandas()

    # Should equal to just querying the original GDF
    expect = gdf.query(expr).to_pandas()
    assert_frame_equal(got, expect)
