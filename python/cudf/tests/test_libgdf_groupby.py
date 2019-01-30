# Copyright (c) 2018, NVIDIA CORPORATION.

import pytest

import numpy as np
import pandas as pd

from cudf.dataframe import DataFrame


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
    # gdf
    got_df = make_frame(DataFrame, nelem=nelem).groupby(
        ['x', 'y'], method="hash").mean()
    got = np.sort(got_df['mean_val'].to_array())
    # pandas
    expect_df = make_frame(pd.DataFrame,
                           nelem=nelem).groupby(['x', 'y']).mean()
    expect = np.sort(expect_df['val'].values)
    # verify
    np.testing.assert_array_almost_equal(expect, got)


@pytest.mark.parametrize('nelem', [2, 3, 100, 1000])
def test_groupby_mean_3level(nelem):
    lvls = 'z'
    bys = list('xyz')
    # gdf
    got_df = make_frame(DataFrame, nelem=nelem, extra_levels=lvls)\
        .groupby(bys, method="hash").mean()
    got = np.sort(got_df['mean_val'].to_array())
    # pandas
    expect_df = make_frame(pd.DataFrame, nelem=nelem,
                           extra_levels=lvls).groupby(bys).mean()
    expect = np.sort(expect_df['val'].values)
    # verify
    np.testing.assert_array_almost_equal(expect, got)


@pytest.mark.parametrize('nelem', [2, 3, 100, 1000])
def test_groupby_agg_mean_min(nelem):
    # gdf (Note: lack of multindex)
    got_df = make_frame(DataFrame, nelem=nelem).groupby(
        ['x', 'y'], method="hash").agg(['mean', 'min'])
    got_mean = np.sort(got_df['mean_val'].to_array())
    got_min = np.sort(got_df['min_val'].to_array())
    # pandas
    expect_df = make_frame(pd.DataFrame, nelem=nelem).groupby(['x', 'y'])\
                                                     .agg(['mean', 'min'])
    expect_mean = np.sort(expect_df['val', 'mean'].values)
    expect_min = np.sort(expect_df['val', 'min'].values)
    # verify
    np.testing.assert_array_almost_equal(expect_mean, got_mean)
    np.testing.assert_array_almost_equal(expect_min, got_min)


@pytest.mark.parametrize('nelem', [2, 3, 100, 1000])
def test_groupby_agg_min_max_dictargs(nelem):
    # gdf (Note: lack of multindex)
    got_df = make_frame(DataFrame, nelem=nelem, extra_vals='ab').groupby(
        ['x', 'y'], method="hash").agg({'a': 'min', 'b': 'max'})
    got_min = np.sort(got_df['min_a'].to_array())
    got_max = np.sort(got_df['max_b'].to_array())
    # pandas
    expect_df = make_frame(pd.DataFrame, nelem=nelem, extra_vals='ab').groupby(
        ['x', 'y']).agg({'a': 'min', 'b': 'max'})
    expect_min = np.sort(expect_df['a'].values)
    expect_max = np.sort(expect_df['b'].values)
    # verify
    np.testing.assert_array_almost_equal(expect_min, got_min)
    np.testing.assert_array_almost_equal(expect_max, got_max)


@pytest.mark.parametrize('nelem', [2, 3, 100, 1000])
def test_groupby_agg_min_max_dictlist(nelem):
    # gdf (Note: lack of multindex)
    got_df = make_frame(DataFrame, nelem=nelem, extra_vals='ab').groupby(
        ['x', 'y'], method="hash").agg({'a': ['min', 'max'],
                                        'b': ['min', 'max']})
    got_min_a = np.sort(got_df['min_a'].to_array())
    got_max_a = np.sort(got_df['max_a'].to_array())
    got_min_b = np.sort(got_df['min_b'].to_array())
    got_max_b = np.sort(got_df['max_b'].to_array())
    # pandas
    expect_df = make_frame(pd.DataFrame, nelem=nelem, extra_vals='ab').groupby(
        ['x', 'y']).agg({'a': ['min', 'max'], 'b': ['min', 'max']})
    expect_min_a = np.sort(expect_df['a']['min'].values)
    expect_max_a = np.sort(expect_df['a']['max'].values)
    expect_min_b = np.sort(expect_df['b']['min'].values)
    expect_max_b = np.sort(expect_df['b']['max'].values)
    # verify
    np.testing.assert_array_almost_equal(expect_min_a, got_min_a)
    np.testing.assert_array_almost_equal(expect_max_a, got_max_a)
    np.testing.assert_array_almost_equal(expect_min_b, got_min_b)
    np.testing.assert_array_almost_equal(expect_max_b, got_max_b)


@pytest.mark.parametrize('nelem', [2, 3, 100, 1000])
@pytest.mark.parametrize('func', ['mean', 'min', 'max', 'count', 'sum'])
def test_groupby_2keys_agg(nelem, func):
    # gdf (Note: lack of multindex)
    got_df = make_frame(DataFrame, nelem=nelem)\
        .groupby(['x', 'y'], method="hash").agg(func)

    got_agg = np.sort(got_df[func + '_val'].to_array())
    # pandas
    expect_df = make_frame(pd.DataFrame, nelem=nelem)\
        .groupby(['x', 'y']).agg(func)

    expect_agg = np.sort(expect_df['val'].values)
    # verify
    np.testing.assert_array_almost_equal(expect_agg, got_agg)
