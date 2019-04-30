# Copyright (c) 2018, NVIDIA CORPORATION.

import pytest

import numpy as np
import pandas as pd

from cudf.dataframe import DataFrame
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


@pytest.mark.parametrize('nelem', [2, 3, 100, 1000])
def test_groupby_mean(nelem):
    got_df = make_frame(DataFrame, nelem=nelem).groupby(
        ['x', 'y'], method="hash").mean()
    expect_df = make_frame(pd.DataFrame,
                           nelem=nelem).groupby(['x', 'y']).mean()
    assert_eq(got_df, expect_df)


@pytest.mark.parametrize('nelem', [2, 3, 100, 1000])
def test_groupby_mean_3level(nelem):
    lvls = 'z'
    bys = list('xyz')
    got_df = make_frame(DataFrame, nelem=nelem, extra_levels=lvls)\
        .groupby(bys, method="hash").mean()
    expect_df = make_frame(pd.DataFrame, nelem=nelem,
                           extra_levels=lvls).groupby(bys).mean()
    assert_eq(got_df, expect_df)


@pytest.mark.parametrize('nelem', [2, 3, 100, 1000])
def test_groupby_agg_mean_min(nelem):
    got_df = make_frame(DataFrame, nelem=nelem).groupby(
        ['x', 'y'], method="hash").agg(['mean', 'min'])
    expect_df = make_frame(pd.DataFrame, nelem=nelem).groupby(['x', 'y'])\
                                                     .agg(['mean', 'min'])
    assert_eq(got_df, expect_df)


@pytest.mark.parametrize('nelem', [2, 3, 100, 1000])
def test_groupby_agg_min_max_dictargs(nelem):
    expect_df = make_frame(pd.DataFrame, nelem=nelem, extra_vals='ab').groupby(
        ['x', 'y']).agg({'a': 'min', 'b': 'max'})
    got_df = make_frame(DataFrame, nelem=nelem, extra_vals='ab').groupby(
        ['x', 'y'], method="hash").agg({'a': 'min', 'b': 'max'})
    assert_eq(expect_df, got_df)


@pytest.mark.parametrize('nelem', [2, 3, 100, 1000])
def test_groupby_agg_min_max_dictlist(nelem):
    expect_df = make_frame(pd.DataFrame, nelem=nelem, extra_vals='ab').groupby(
        ['x', 'y']).agg({'a': ['min', 'max'], 'b': ['min', 'max']})
    got_df = make_frame(DataFrame, nelem=nelem, extra_vals='ab').groupby(
        ['x', 'y'], method="hash").agg({'a': ['min', 'max'],
                                        'b': ['min', 'max']})
    assert_eq(got_df, expect_df)


@pytest.mark.parametrize('nelem', [2, 3, 100, 1000])
@pytest.mark.parametrize('func', ['mean', 'min', 'max', 'count', 'sum'])
def test_groupby_2keys_agg(nelem, func):
    # gdf (Note: lack of multindex)
    expect_df = make_frame(pd.DataFrame, nelem=nelem)\
        .groupby(['x', 'y']).agg(func)
    got_df = make_frame(DataFrame, nelem=nelem)\
        .groupby(['x', 'y'], method="hash").agg(func)
    assert_eq(got_df, expect_df)
