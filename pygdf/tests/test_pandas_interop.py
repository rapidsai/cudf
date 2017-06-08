import numpy as np
import pandas as pd

from pygdf.dataframe import DataFrame


def test_to_pandas():
    df = DataFrame()
    df['a'] = np.arange(10, dtype=np.int32)
    df['b'] = np.arange(10, 20, dtype=np.float64)

    pdf = df.to_pandas()

    assert len(df.columns) == len(pdf.columns)

    assert df['a'].dtype == pdf['a'].dtype
    assert df['b'].dtype == pdf['b'].dtype

    assert len(df['a']) == len(pdf['a'])
    assert len(df['b']) == len(pdf['b'])


def test_from_pandas():
    pdf = pd.DataFrame()
    pdf['a'] = np.arange(10, dtype=np.int32)
    pdf['b'] = np.arange(10, 20, dtype=np.float64)

    df = DataFrame.from_pandas(pdf)

    assert len(df.columns) == len(pdf.columns)

    assert df['a'].dtype == pdf['a'].dtype
    assert df['b'].dtype == pdf['b'].dtype

    assert len(df['a']) == len(pdf['a'])
    assert len(df['b']) == len(pdf['b'])
