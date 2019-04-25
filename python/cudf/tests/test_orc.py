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

    expect = orcfile.read(columns=columns).to_pandas(date_as_object=False)
    got = cudf.read_orc(path, engine=engine, columns=columns)

    # cuDF's default currently handles some types differently
    if engine == 'cudf':
        # cuDF doesn't support bool so convert to int8
        if 'boolean1' in expect.columns:
            expect['boolean1'] = expect['boolean1'].astype('int8')

    assert_eq(expect, got, check_categorical=False)


@pytest.mark.filterwarnings("ignore:Strings are not yet supported")
def test_orc_reader_decimal(datadir):
    path = datadir / 'TestOrcFile.decimal.orc'
    orcfile = pa.orc.ORCFile(path)

    pdf = orcfile.read().to_pandas()
    gdf = cudf.read_orc(path, engine='cudf').to_pandas()

    # Convert the decimal dtype from PyArrow to float64 for comparison to cuDF
    # This is because cuDF returns as float64 as it lacks equivalent dtype
    pdf = pdf.apply(pd.to_numeric)

    if '_col0' in gdf.columns:
        expectcol = pdf['_col0']
        gotcol = gdf['_col0']
        for i in range(len(gotcol)):
            if expectcol[i] != gotcol[i]:
                print("Time mismatched at [", i, "] expect: ",
                      expectcol[i], " got: ", gotcol[i])
                break

    np.testing.assert_allclose(pdf, gdf)
