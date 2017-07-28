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
    print(got_df)
    got = np.sort(got_df['val'].to_array())
    # pandas
    expect_df = make_frame(pd.DataFrame, nelem=nelem).groupby(('x', 'y')).mean()
    expect = np.sort(expect_df['val'].values)
    print(expect_df)
    # verify
    np.testing.assert_array_almost_equal(expect, got)

