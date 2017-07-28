import numpy as np
import pandas as pd

from pygdf.dataframe import DataFrame


def make_frame(dataframe_class, nelem=20, seed=0):
    np.random.seed(seed)

    df = dataframe_class()

    df['x'] = np.random.randint(0, 5, nelem)
    df['y'] = np.random.randint(0, 3, nelem)
    df['val'] = np.random.random(nelem)

    return df


def test_groupby_mean():
    # gdf
    got_df = make_frame(DataFrame).groupby(('x', 'y')).mean()
    got = got_df['val'].to_array()
    # pandas
    expect_df = make_frame(pd.DataFrame).groupby(('x', 'y')).mean()
    expect = expect_df['val'].values
    # verify
    np.testing.assert_array_almost_equal(expect, got)

