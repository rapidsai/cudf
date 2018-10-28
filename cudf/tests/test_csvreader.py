# Copyright (c) 2018, NVIDIA CORPORATION.

import pytest
from collections import OrderedDict

import numpy as np
import pandas as pd

from cudf import read_csv


def make_numeric_dataframe(nrows, dtype):
    df = pd.DataFrame()
    df['col1'] = np.arange(nrows, dtype=dtype)
    df['col2'] = np.arange(1, 1 + nrows, dtype=dtype)
    return df


def make_datetime_dataframe():
    df = pd.DataFrame()
    df['col1'] = np.array(['31/10/2010', '05/03/2001', '20/10/1994',
                          '18/10/1990'])
    df['col2'] = np.array(['18/04/1995', '14/07/1994', '07/06/2006',
                          '16/09/2005'])
    return df


def make_numpy_mixed_dataframe():
    df = pd.DataFrame()
    df['Integer'] = np.array([2345, 11987, 9027, 53916])
    df['Date'] = np.array(['18/04/1995', '14/07/1994', '07/06/2006',
                          '16/09/2005'])
    df['Float'] = np.array([9.001, 8.343, 6, 2.781])
    df['Category'] = np.array(['M', 'F', 'F', 'F'])
    return df


def make_all_numeric_dtypes_dataframe():
    df = pd.DataFrame()

    gdf_dtypes = ["float", "float32", "float64", "double", "short", "int",
                  "int32", "int64", "long"]

    np_dtypes = [np.float32, np.float32, np.float64, np.float64, np.int16,
                 np.int32, np.int32, np.int64, np.int64]

    for i in range(len(gdf_dtypes)):
        df[gdf_dtypes[i]] = np.arange(10, dtype=np_dtypes[i])

    return (
        df,
        OrderedDict(zip(gdf_dtypes, gdf_dtypes)),
        OrderedDict(zip(gdf_dtypes, np_dtypes))
    )


dtypes = [np.float64, np.float32, np.int64, np.int32]
dtypes_dict = {'1': np.float64, '2': np.float32, '3': np.int64, '4': np.int32}
nelem = [5, 25, 100]


@pytest.mark.parametrize('dtype', dtypes)
@pytest.mark.parametrize('nelem', nelem)
def test_csv_reader_numeric_data(dtype, nelem, tmpdir):

    fname = tmpdir.mkdir("gdf_csv").join("tmp_csvreader_file1.csv")

    df = make_numeric_dataframe(nelem, dtype)
    df.to_csv(fname, index=False, header=False)

    dtypes = [df[k].dtype for k in df.columns]
    out = read_csv(str(fname), names=list(df.columns.values), dtype=dtypes)

    assert len(out.columns) == len(df.columns)
    pd.util.testing.assert_frame_equal(df, out.to_pandas())


def test_csv_reader_datetime_data(tmpdir):

    fname = tmpdir.mkdir("gdf_csv").join('tmp_csvreader_file2.csv')

    df = make_datetime_dataframe()
    df.to_csv(fname, index=False, header=False)

    df_out = pd.read_csv(fname, names=['col1', 'col2'], parse_dates=[0, 1],
                         dayfirst=True)
    dtypes = ['date', 'date']
    out = read_csv(str(fname), names=list(df.columns.values), dtype=dtypes,
                   dayfirst=True)

    assert len(out.columns) == len(df_out.columns)
    pd.util.testing.assert_frame_equal(df_out, out.to_pandas())


def test_csv_reader_mixed_data_delimiter(tmpdir):

    fname = tmpdir.mkdir("gdf_csv").join('tmp_csvreader_file3.csv')

    df = make_numpy_mixed_dataframe()
    df.to_csv(fname, sep='|', index=False, header=False)

    out = read_csv(str(fname), delimiter='|', names=['1', '2', '3', '4'],
                   dtype=['int64', 'date', 'float64', 'category'],
                   dayfirst=True)
    df_out = pd.read_csv(fname, delimiter='|', names=['1', '2', '3', '4'],
                         parse_dates=[1], dayfirst=True)

    assert len(out.columns) == len(df_out.columns)


def test_csv_reader_all_numeric_dtypes(tmpdir):

    # fname = os.path.abspath('cudf/tests/data/tmp_csvreader_file4.csv')
    fname = tmpdir.mkdir("gdf_csv").join("tmp_csvreader_file4.csv")

    df, gdf_dict, pd_dict = make_all_numeric_dtypes_dataframe()
    df.to_csv(fname, sep=',', index=False, header=False)

    out = read_csv(str(fname), delimiter=',', names=list(gdf_dict.keys()),
                   dtype=gdf_dict)
    df_out = pd.read_csv(fname, delimiter=',', names=list(pd_dict.keys()),
                         dtype=pd_dict, dayfirst=True)

    assert len(out.columns) == len(df_out.columns)
    pd.util.testing.assert_frame_equal(df_out, out.to_pandas())


def test_csv_reader_skiprows_skipfooter(tmpdir):

    fname = tmpdir.mkdir("gdf_csv").join("tmp_csvreader_file5.csv")

    df = make_numpy_mixed_dataframe()
    df.to_csv(fname, columns=['Integer', 'Date', 'Float'], index=False,
              header=False)

    # Using engine='python' to eliminate pandas warning of using python engine.
    df_out = pd.read_csv(fname, names=['1', '2', '3'], parse_dates=[1],
                         dayfirst=True, skiprows=1, skipfooter=1,
                         engine='python')
    out = read_csv(str(fname), names=['1', '2', '3'],
                   dtype=['int64', 'date', 'float64'], skiprows=1,
                   skipfooter=1, dayfirst=True)

    assert len(out.columns) == len(df_out.columns)
    assert len(out) == len(df_out)
    pd.util.testing.assert_frame_equal(df_out, out.to_pandas())
