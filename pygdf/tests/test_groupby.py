import pytest

import numpy as np
import pandas as pd

from pygdf.dataframe import DataFrame


def make_frame(dataframe_class, nelem, seed=0):
    np.random.seed(seed)

    df = dataframe_class()

    df['x'] = np.random.randint(0, 5, nelem)
    df['y'] = np.random.randint(0, 3, nelem)
    df['val'] = np.random.random(nelem)

    return df


@pytest.mark.parametrize('nelem', [2, 3, 100, 1000])
def test_groupby_mean(nelem):
    # gdf
    got_df = make_frame(DataFrame, nelem=nelem).groupby(('x', 'y')).mean()
    got = np.sort(got_df['val'].to_array())
    # pandas
    expect_df = make_frame(pd.DataFrame, nelem=nelem).groupby(('x', 'y')).mean()
    expect = np.sort(expect_df['val'].values)
    # verify
    np.testing.assert_array_almost_equal(expect, got)


@pytest.mark.parametrize('nelem', [2, 100])
def test_groupby_agg_mean_min(nelem):
    # gdf (Note: lack of multindex)
    got_df = make_frame(DataFrame, nelem=nelem).groupby(('x', 'y')).agg(['mean', 'min'])
    got_mean = np.sort(got_df['val_mean'].to_array())
    got_min = np.sort(got_df['val_min'].to_array())
    # pandas
    expect_df = make_frame(pd.DataFrame, nelem=nelem).groupby(('x', 'y')).agg(['mean', 'min'])
    expect_mean = np.sort(expect_df['val', 'mean'].values)
    expect_min = np.sort(expect_df['val', 'min'].values)
    # verify
    np.testing.assert_array_almost_equal(expect_mean, got_mean)
    np.testing.assert_array_almost_equal(expect_min, got_min)


def test_groupby_cats():
    df = DataFrame()
    df['cats'] = pd.Categorical(list('aabaacaab'))
    df['vals'] = np.random.random(len(df))

    cats = np.asarray(list(df['cats']))
    vals = df['vals'].to_array()

    grouped = df.groupby(['cats']).mean()

    got_vals = grouped['vals']
    got_cats = grouped['cats']

    for c, v in zip(got_cats, got_vals):
        print(c, v)
        expect = vals[cats == c].mean()
        np.testing.assert_almost_equal(v, expect)

