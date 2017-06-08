import numpy as np

from pygdf.dataframe import DataFrame


def test_to_pandas():
    df = DataFrame()
    df['a'] = np.arange(10, dtype=np.int32)
    df['b'] = np.arange(10, 20, dtype=np.float64)

    pdf = df.to_pandas()

    assert df['a'].dtype == pdf['a'].dtype
    assert df['b'].dtype == pdf['b'].dtype

    assert len(df['a']) == len(pdf['a'])
    assert len(df['b']) == len(pdf['b'])
