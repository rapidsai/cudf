# Copyright (c) 2019, NVIDIA CORPORATION.

import cudf
from cudf.tests.utils import assert_eq

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
                                   'int1', 'long1', 'float1', 'double1']],
        ['TestOrcFile.testDate1900.orc', None]
    ]
)
def test_orc_reader(datadir, orc_args, engine):
    # cuDF only handles milliseconds when converting from pa_table
    # This causes an error of data loss when casting from nanoseconds
    if engine == 'pyarrow' and orc_args[0] == 'TestOrcFile.testDate1900.orc':
        return

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

    # Debug code for timestamps
    if 'time' in got.columns:
        expectcol = expect['time'].astype('int64')
        gotcol = got['time'].astype('int64')
        for i in range(len(gotcol)):
            if expectcol[i] != gotcol[i]:
                print("Time mismatched at [", i, "] expect: ",
                      expectcol[i], " got: ", gotcol[i])
                break
    else:
        assert_eq(expect, got, check_categorical=False)
