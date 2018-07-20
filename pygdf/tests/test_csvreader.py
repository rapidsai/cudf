import os.path
import pytest

import numpy as np
import pandas as pd

from numba import cuda

from pygdf.dataframe import Series, DataFrame
from pygdf.io import read_csv
from pygdf._gdf import libgdf, ffi, _as_numba_devarray

def make_dataframe(nrows):
    df = pd.DataFrame()
    df['col1'] = np.arange(nrows)
    df['colTwo'] = np.arange(1, 1 + nrows)
    return df


def test_csv_reader():
    print(cuda.current_context())  # XXX: read_csv seems to use a different context

    fname = os.path.abspath('tmp_csvreader_file.csv')
    print(fname)
    df = make_dataframe(nrows=1000)
    df.to_csv(fname, index=False, header=False)

    with open(fname, 'r') as fin:
        print(fin.read())

    dtypes = [df[k].dtype for k in df.columns]
    print(df.columns)
    print('list(df.columns)', list(df.columns.values))
    print(list(map(str, dtypes)))
    gdf = read_csv(fname, names=list(df.columns.values), dtypes=dtypes)
    cuda.synchronize()
    print(gdf)
    print(gdf[0].data)
    print(gdf[0].valid)
    print(gdf[0].dtype)
    print(gdf[0].size)

    intptr = int(ffi.cast('intptr_t', gdf[0].data))
    da = _as_numba_devarray(intptr, gdf[0].size, dtype=np.int64)
    print(da[0])
    print(da.copy_to_host())

    assert gdf[0].dtype == libgdf.GDF_INT64



