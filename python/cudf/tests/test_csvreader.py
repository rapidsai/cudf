# Copyright (c) 2018, NVIDIA CORPORATION.

import pytest
from collections import OrderedDict

import numpy as np
import pandas as pd

from cudf import read_csv
from cudf.io.csv import read_csv_strings
import cudf
import nvstrings


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
    df['Integer'] = np.array([2345, 11987, 9027, 9027])
    df['Date'] = np.array(['18/04/1995', '14/07/1994', '07/06/2006',
                          '16/09/2005'])
    df['Float'] = np.array([9.001, 8.343, 6, 2.781])
    df['Integer2'] = np.array([2345, 106, 2088, 789277])
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

    out = read_csv(str(fname), delimiter='|', names=['1', '2', '3', '4', '5'],
                   dtype=['int64', 'date', 'float64', 'int64', 'category'],
                   dayfirst=True)
    df_out = pd.read_csv(fname, delimiter='|', names=['1', '2', '3', '4', '5'],
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


def test_csv_reader_negative_vals(tmpdir):
    fname = tmpdir.mkdir("gdf_csv").join("tmp_csvreader_file6.csv")

    names = ['0', '1', '2']
    dtypes = ['float32', 'float32', 'float32']
    lines = [','.join(names),
             '-181.5060,-185.37000,-3',
             '-127.6300,-230.54600,-9']

    with open(str(fname), 'w') as fp:
        fp.write('\n'.join(lines) + '\n')

    zero = [-181.5060, -127.6300]
    one = [-185.370, -230.54600]
    two = [-3, -9]

    df = read_csv(str(fname), names=names, dtype=dtypes, skiprows=1)

    np.testing.assert_allclose(zero, df['0'])
    np.testing.assert_allclose(one, df['1'])
    np.testing.assert_allclose(two, df['2'])


def test_csv_reader_strings(tmpdir):
    fname = tmpdir.mkdir("gdf_csv").join("tmp_csvreader_file7.csv")

    names = ['text', 'int']
    dtypes = ['str', 'int']
    lines = [','.join(names), 'a,0', 'b,0', 'c,0', 'd,0']

    with open(str(fname), 'w') as fp:
        fp.write('\n'.join(lines) + '\n')

    cols = read_csv_strings(str(fname), names=names, dtype=dtypes, skiprows=1)

    assert(len(cols) == 2)
    assert(type(cols[0]) == nvstrings.nvstrings)
    assert(type(cols[1]) == cudf.Series)
    assert(cols[0].sublist([0]).to_host()[0] == 'a')
    assert(cols[0].sublist([1]).to_host()[0] == 'b')
    assert(cols[0].sublist([2]).to_host()[0] == 'c')
    assert(cols[0].sublist([3]).to_host()[0] == 'd')


def test_csv_reader_strings_quotechars(tmpdir):
    fname = tmpdir.mkdir("gdf_csv").join("tmp_csvreader_file8.csv")

    names = ['text', 'int']
    dtypes = ['str', 'int']
    lines = [','.join(names), '"a,\n",0', '"b ""c"" d",0', 'e,0', '"f,,!.,",0']

    with open(str(fname), 'w') as fp:
        fp.write('\n'.join(lines) + '\n')

    cols = read_csv_strings(str(fname), names=names, dtype=dtypes, skiprows=1,
                            quotechar='\"', quoting=True)

    assert(len(cols) == 2)
    assert(type(cols[0]) == nvstrings.nvstrings)
    assert(type(cols[1]) == cudf.Series)
    assert(cols[0].sublist([0]).to_host()[0] == 'a,\n')
    assert(cols[0].sublist([1]).to_host()[0] == 'b "c" d')
    assert(cols[0].sublist([2]).to_host()[0] == 'e')
    assert(cols[0].sublist([3]).to_host()[0] == 'f,,!.,')


def test_csv_reader_auto_column_detection(tmpdir):
    fname = tmpdir.mkdir("gdf_csv").join("tmp_csvreader_file7.csv")
    df = make_numpy_mixed_dataframe()
    df.to_csv(fname, columns=['Integer', 'Date', 'Float'], index=False,
              header=False)

    df_out = pd.read_csv(fname, parse_dates=[1], dayfirst=True)
    out = read_csv(str(fname), dayfirst=True)
    assert len(out.columns) == len(df_out.columns)
    assert len(out) == len(df_out)
    pd.util.testing.assert_frame_equal(df_out, out.to_pandas())
    # Check dtypes
    assert list(df_out.dtypes) == list(out.to_pandas().dtypes)


def test_csv_reader_usecols_int_char(tmpdir):
    fname = tmpdir.mkdir("gdf_csv").join("tmp_csvreader_file7.csv")
    df = make_numpy_mixed_dataframe()
    df.to_csv(fname, columns=['Integer', 'Date', 'Float', 'Integer2'],
              index=False, header=False)

    df_out = pd.read_csv(fname, usecols=[0, 1, 3], parse_dates=[1],
                         dayfirst=True)
    out = read_csv(str(fname), usecols=[0, 1, 3], dayfirst=True)
    print(df_out)
    print(out)
    assert len(out.columns) == len(df_out.columns)
    assert len(out) == len(df_out)
    pd.util.testing.assert_frame_equal(df_out, out.to_pandas(),
                                       check_names=False)


def test_csv_reader_mangle_dupe_cols_header(tmpdir):
    fname = tmpdir.mkdir("gdf_csv").join("tmp_csvreader_file7.csv")
    df = make_numpy_mixed_dataframe()
    df.to_csv(fname, columns=['Integer', 'Date', 'Float', 'Integer2'],
              index=False, header=False)

    # Default: header=0 when names not passed, mangle_dupe_cols = True
    df_out = pd.read_csv(fname, parse_dates=[1], dayfirst=True)
    out = read_csv(str(fname), dayfirst=True)
    assert len(out.columns) == len(df_out.columns)
    assert len(out) == len(df_out)
    # Compare mangled column names for duplicate names in header row
    assert list(df_out.columns.values) == list(out.columns.values)
    pd.util.testing.assert_frame_equal(df_out, out.to_pandas())

    # header = 3
    df_out = pd.read_csv(fname, parse_dates=[1], dayfirst=True, header=2)
    out = read_csv(str(fname), dayfirst=True, header=2)
    assert len(out.columns) == len(df_out.columns)
    # assert len(out) == len(df_out)
    # Compare column names
    assert list(df_out.columns.values) == list(out.columns.values)
