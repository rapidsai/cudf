# Copyright (c) 2018, NVIDIA CORPORATION.

import pytest
from collections import OrderedDict

import numpy as np
import pandas as pd

from io import StringIO

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


def test_csv_reader_float_decimal(tmpdir):
    fname = tmpdir.mkdir("gdf_csv").join("tmp_csvreader_file9.csv")

    names = ['basic_32', 'basic_64', 'round', 'decimal_only']
    dtypes = ['float32', 'float64', 'float64', 'float32']
    lines = [';'.join(names),
             '1,2;1234,5678;12345;0,123',
             '3,4;3456,7890;67890;,456']

    with open(str(fname), 'w') as fp:
        fp.write('\n'.join(lines) + '\n')

    basic_32_ref = [1.2, 3.4]
    basic_64_ref = [1234.5678, 3456.7890]
    round_ref = [12345, 67890]
    decimal_only_ref = [0.123, 0.456]

    df = read_csv(str(fname), names=names, dtype=dtypes, skiprows=1,
                  delimiter=';', decimal=',')

    np.testing.assert_allclose(basic_32_ref, df['basic_32'])
    np.testing.assert_allclose(basic_64_ref, df['basic_64'])
    np.testing.assert_allclose(round_ref, df['round'])
    np.testing.assert_allclose(decimal_only_ref, df['decimal_only'])


def test_csv_reader_thousands(tmpdir):
    fname = tmpdir.mkdir("gdf_csv").join("tmp_csvreader_file10.csv")

    names = dtypes = ["float32", "float64", "int32", "int64"]
    lines = [','.join(names),
             "1'234.5, 1'234.567, 1'234'567, 1'234'567'890",
             "12'345.6, 123'456.7, 12'345, 123'456'789"]

    with open(str(fname), 'w') as fp:
        fp.write('\n'.join(lines) + '\n')

    f32_ref = [1234.5, 12345.6]
    f64_ref = [1234.567, 123456.7]
    int32_ref = [1234567, 12345]
    int64_ref = [1234567890, 123456789]

    df = read_csv(str(fname), names=names, dtype=dtypes, skiprows=1,
                  thousands="'")

    np.testing.assert_allclose(f32_ref, df['float32'])
    np.testing.assert_allclose(f64_ref, df['float64'])
    np.testing.assert_allclose(int32_ref, df['int32'])
    np.testing.assert_allclose(int64_ref, df['int64'])


def test_csv_reader_buffer(tmpdir):

    names = dtypes = ["float32", "int32", "date"]
    lines = [','.join(names),
             "1234.5, 1234567, 11/22/1995",
             "12345.6, 12345, 1/2/2002"]

    buffer = '\n'.join(lines) + '\n'

    f32_ref = [1234.5, 12345.6]
    int32_ref = [1234567, 12345]

    df = read_csv(StringIO(buffer), names=names, dtype=dtypes, skiprows=1)

    np.testing.assert_allclose(f32_ref, df['float32'])
    np.testing.assert_allclose(int32_ref, df['int32'])
    assert("1995-11-22T00:00:00.000" == str(df['date'][0]))
    assert("2002-01-02T00:00:00.000" == str(df['date'][1]))

def test_csv_reader_buffer_strings(tmpdir):

    names = ['text', 'int']
    dtypes = ['str', 'int']
    lines = [','.join(names), 'a,0', 'b,0', 'c,0', 'd,0']

    buffer = '\n'.join(lines) + '\n'

    cols = read_csv_strings(StringIO(buffer), names=names, dtype=dtypes, skiprows=1)

    assert(len(cols) == 2)
    assert(type(cols[0]) == nvstrings.nvstrings)
    assert(type(cols[1]) == cudf.Series)
    assert(cols[0].sublist([0]).to_host()[0] == 'a')
    assert(cols[0].sublist([1]).to_host()[0] == 'b')
    assert(cols[0].sublist([2]).to_host()[0] == 'c')
    assert(cols[0].sublist([3]).to_host()[0] == 'd')