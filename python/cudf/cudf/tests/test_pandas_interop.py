# Copyright (c) 2018, NVIDIA CORPORATION.

import numpy as np
import pandas as pd

from cudf.dataframe import DataFrame


def test_to_pandas():
    df = DataFrame()
    df['a'] = np.arange(5, dtype=np.int32)
    df['b'] = np.arange(10, 15, dtype=np.float64)
    df['c'] = np.array([True, False, None, True, True])

    pdf = df.to_pandas()

    assert tuple(df.columns) == tuple(pdf.columns)

    assert df['a'].dtype == pdf['a'].dtype
    assert df['b'].dtype == pdf['b'].dtype

    # Notice, the dtype differ when Pandas and cudf boolean series
    # contains None/NaN
    assert df['c'].dtype == np.bool
    assert pdf['c'].dtype == np.object

    assert len(df['a']) == len(pdf['a'])
    assert len(df['b']) == len(pdf['b'])
    assert len(df['c']) == len(pdf['c'])


def test_from_pandas():
    pdf = pd.DataFrame()
    pdf['a'] = np.arange(10, dtype=np.int32)
    pdf['b'] = np.arange(10, 20, dtype=np.float64)

    df = DataFrame.from_pandas(pdf)

    assert tuple(df.columns) == tuple(pdf.columns)

    assert df['a'].dtype == pdf['a'].dtype
    assert df['b'].dtype == pdf['b'].dtype

    assert len(df['a']) == len(pdf['a'])
    assert len(df['b']) == len(pdf['b'])


def test_from_pandas_ex1():
    pdf = pd.DataFrame({'a': [0, 1, 2, 3],
                        'b': [0.1, 0.2, None, 0.3]})
    print(pdf)
    df = DataFrame.from_pandas(pdf)
    print(df)

    assert tuple(df.columns) == tuple(pdf.columns)
    assert np.all(df['a'].to_array() == pdf['a'])
    matches = df['b'].to_array(fillna='pandas') == pdf['b']
    # the 3d element is False due to (nan == nan) == False
    assert np.all(matches == [True, True, False, True])
    assert np.isnan(df['b'].to_array(fillna='pandas')[2])
    assert np.isnan(pdf['b'][2])


def test_from_pandas_with_index():
    pdf = pd.DataFrame({'a': [0, 1, 2, 3],
                        'b': [0.1, 0.2, None, 0.3]})
    pdf = pdf.set_index(np.asarray([4, 3, 2, 1]))
    df = DataFrame.from_pandas(pdf)

    # Check columns
    np.testing.assert_array_equal(df.a.to_array(fillna='pandas'), pdf.a)
    np.testing.assert_array_equal(df.b.to_array(fillna='pandas'), pdf.b)
    # Check index
    np.testing.assert_array_equal(df.index.values, pdf.index.values)
    # Check again using pandas testing tool on frames
    pd.util.testing.assert_frame_equal(df.to_pandas(), pdf)
