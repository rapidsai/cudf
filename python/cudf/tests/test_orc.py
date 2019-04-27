# Copyright (c) 2019, NVIDIA CORPORATION.

import cudf
from cudf.tests.utils import assert_eq

import numpy as np
import pandas as pd
import pytest
import pyarrow as pa


@pytest.fixture(scope='module')
def datadir(datadir):
    return datadir / 'orc'


@pytest.mark.filterwarnings("ignore:Using CPU")
@pytest.mark.filterwarnings("ignore:Strings are not yet supported")
@pytest.mark.parametrize('engine', ['pyarrow', 'cudf'])
@pytest.mark.parametrize(
    'orc_args',
    [
        ['TestOrcFile.emptyFile.orc', ['boolean1']],
        ['TestOrcFile.test1.orc', ['boolean1', 'byte1', 'short1',
                                   'int1', 'long1', 'float1', 'double1']]
    ]
)
def test_orc_reader_basic(datadir, orc_args, engine):
    path = datadir / orc_args[0]
    orcfile = pa.orc.ORCFile(path)
    columns = orc_args[1]

    expect = orcfile.read(columns=columns).to_pandas()
    got = cudf.read_orc(path, engine=engine, columns=columns)

    # cuDF's default currently handles some types differently
    if engine == 'cudf':
        # cuDF doesn't support bool so convert to int8
        if 'boolean1' in expect.columns:
            expect['boolean1'] = expect['boolean1'].astype('int8')

    assert_eq(expect, got, check_categorical=False)


def test_orc_reader_decimal(datadir):
    path = datadir / 'TestOrcFile.decimal.orc'
    orcfile = pa.orc.ORCFile(path)

    pdf = orcfile.read().to_pandas()
    gdf = cudf.read_orc(path, engine='cudf').to_pandas()

    # Convert the decimal dtype from PyArrow to float64 for comparison to cuDF
    # This is because cuDF returns as float64 as it lacks equivalent dtype
    pdf = pdf.apply(pd.to_numeric)

    np.testing.assert_allclose(pdf, gdf)


@pytest.mark.parametrize('inputfile', ['TestOrcFile.testDate1900.orc',
                                       'TestOrcFile.testDate2038.orc'])
def test_orc_reader_datetimestamp(datadir, inputfile):
    path = datadir / inputfile
    orcfile = pa.orc.ORCFile(path)

    pdf = orcfile.read().to_pandas(date_as_object=False)
    gdf = cudf.read_orc(path, engine='cudf')

    # cuDF DatetimeColumn currenly only supports millisecond units
    # Convert to lesser precision int64 for comparison
    timedelta = np.timedelta64(1, 'ms').astype('timedelta64[ns]')
    pdf['time'] = pdf['time'].astype(np.int64) // timedelta.astype(np.int64)
    gdf['time'] = gdf['time'].astype(np.int64)

    assert_eq(pdf, gdf, check_categorical=False)
