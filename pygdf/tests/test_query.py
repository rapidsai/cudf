from __future__ import print_function, division

import inspect

import pytest
import numpy as np
from itertools import product

from pygdf import queryutils
from pygdf.dataframe import DataFrame


_params_query_parser = []
_params_query_parser.append(('a > @b', ('a', '__PYGDF_ENVREF__b')))
_params_query_parser.append(('(a + b) <= @c', ('a', 'b', '__PYGDF_ENVREF__c')))
_params_query_parser.append(('a > b if a > 0 else b > a', ('a', 'b')))


@pytest.mark.parametrize('text,expect_args', _params_query_parser)
def test_query_parser(text, expect_args):
    info = queryutils.query_parser(text)
    fn = queryutils.query_builder(info, 'myfoo')
    assert callable(fn)
    argspec = inspect.getargspec(fn)
    assert tuple(argspec.args) == tuple(expect_args)


params_query_data = list(product([7, 8, 9, 16, 100, 129], range(2)))
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


