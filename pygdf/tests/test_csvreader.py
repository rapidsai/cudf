import os.path
import pytest

import numpy as np
import pandas as pd

from pygdf.io import read_csv


def make_dataframe(nrows):
    df = pd.DataFrame()
    df['col1'] = np.arange(nrows)
    df['colTwo'] = np.arange(1, 1 + nrows)
    return df


@pytest.mark.parametrize('nrows', [1, 5, 10, 100])
def test_csv_reader(nrows):
    # XXX: use temp file instead
    fname = os.path.abspath('tmp_csvreader_file.csv')
    print(fname)
    df = make_dataframe(nrows=10)
    df.to_csv(fname, index=False, header=False)

    with open(fname, 'r') as fin:
        print(fin.read())

    dtypes = [df[k].dtype for k in df.columns]
    out = read_csv(fname, names=list(df.columns.values), dtypes=dtypes)
    assert len(out.columns) == len(df.columns)
    pd.util.testing.assert_frame_equal(df, out.to_pandas())
