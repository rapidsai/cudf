import pytest

import numpy as np
import pandas as pd

from pygdf.dataframe import Series, DataFrame
from pygdf.io import read_csv


def make_dataframe(nrows):
    df = pd.DataFrame([
        ('col1', np.arange(nrows)),
        ('colTwo', 1 / np.arange(1, 1 + nrows)),
    ])
    return df


def test_csv_reader():
    fname = 'tmp_csvreader_file'
    df = make_dataframe(nrows=10)
    df.to_csv(fname)

    gdf = read_csv(fname)

    print(type(gdf))
    print(gdf.to_pandas())

