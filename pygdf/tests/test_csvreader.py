import os.path
import pytest

import numpy as np
import pandas as pd

from pygdf import read_csv


def make_numeric_dataframe(nrows, dtype):
    df = pd.DataFrame()
    df['col1'] = np.arange(nrows, dtype=dtype)
    df['col2'] = np.arange(1, 1 + nrows, dtype=dtype)
    return df


def make_datetime_dataframe():
    df = pd.DataFrame()
    df['col1'] = np.array(['31/10/2010', '3/2001', '20/10/1994', '18/10/1990'])
    df['col2'] = np.array(['18/4/1995', '14/7/1994', '7/06/2006', '9/2005'])
    return df


def make_mixed_dataframe():
    df = pd.DataFrame()
    df['Integer'] = np.array([2345, 11987, 9027, 53916])
    df['Date'] = np.array(['18/4/1995', '14/7/1994', '7/06/2006', '9/2005'])
    df['Float'] = np.array([9.001, 8.343, 6, 2.781])
    df['Category'] = np.array(['M', 'F', 'F', 'F'])
    return df


_dtypes = [np.int32, np.int64, np.float32, np.float64]


@pytest.mark.parametrize('dtype', _dtypes)
def test_csv_reader_numericdata(dtype):

    fname = os.path.abspath('tmp_csvreader_file.csv')

    df = make_numeric_dataframe(100, dtype)
    df.to_csv(fname, index=False, header=False)

    dtypes = [df[k].dtype for k in df.columns]
    out = read_csv(fname, names=list(df.columns.values), dtype=dtypes)

    assert len(out.columns) == len(df.columns)
    pd.util.testing.assert_frame_equal(df, out.to_pandas())


def test_csv_reader_datetimedata():

    fname = os.path.abspath('tmp_csvreader_file.csv')

    df = make_datetime_dataframe()
    df.to_csv(fname, index=False, header=False)

    df_out = pd.read_csv(fname, names=['col1', 'col2'], parse_dates=[0, 1],
                         dayfirst=True)
    dtypes = ['date', 'date']
    out = read_csv(fname, names=list(df.columns.values), dtype=dtypes)

    assert len(out.columns) == len(df_out.columns)
    pd.util.testing.assert_frame_equal(df_out, out.to_pandas())


def test_csv_reader_mixeddata_delimiter():

    fname = os.path.abspath('tmp_csvreader_file.csv')

    df = make_mixed_dataframe()
    df.to_csv(fname, sep='|', index=False, header=False)

    out = read_csv(fname, delimiter='|', names=['1', '2', '3', '4'],
                   dtype=['int64', 'date', 'float64', 'category'])
    df_out = pd.read_csv(fname, delimiter='|', names=['1', '2', '3', '4'],
                         parse_dates=[1], dayfirst=True)

    print(out.to_pandas())
    print(df_out)
    assert len(out.columns) == len(df_out.columns)
    assert len(out) == len(df_out)


def test_csv_reader_skiprows_skipfooter():

    fname = os.path.abspath('tmp_csvreader_file.csv')

    df = make_mixed_dataframe()
    df.to_csv(fname, columns=['Integer', 'Date', 'Float'], index=False,
              header=False)

    df_out = pd.read_csv(fname, names=['1', '2', '3'], parse_dates=[1],
                         dayfirst=True, skiprows=1, skipfooter=1)
    out = read_csv(fname, names=['1', '2', '3'],
                   dtype=['int64', 'date', 'float64'], skiprows=1, skipfooter=1
                   )

    assert len(out.columns) == len(df_out.columns)
    assert len(out) == len(df_out)
    pd.util.testing.assert_frame_equal(df_out, out.to_pandas())
