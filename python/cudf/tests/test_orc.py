# Copyright (c) 2019, NVIDIA CORPORATION.

import cudf
from cudf.tests.utils import assert_eq

import numpy as np
import pytest
import pyarrow as pa


@pytest.fixture(scope='module')
def datadir(datadir):
    return datadir / 'orc'


@pytest.mark.filterwarnings("ignore:Using CPU")
@pytest.mark.filterwarnings("ignore:Strings are not yet supported")
@pytest.mark.parametrize('engine', ['cudf'])
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
    path = datadir / orc_args[0]
    orcfile = pa.orc.ORCFile(path)
    columns = orc_args[1]

    expect = orcfile.read(columns=columns).to_pandas(date_as_object=False)
    got = cudf.read_orc(path, engine=engine, columns=columns)

    # cuDF's default currently handles some types differently
    if engine == 'cudf':
        # For bool, cuDF doesn't support it so convert it to int8
        if 'boolean1' in expect.columns:
            expect['boolean1'] = expect['boolean1'].astype('int8')

        # For debug testing, view it as raw nanoseconds
        if 'time' in expect.columns:
            expect['time'] = expect['time'].astype('int64')
        if 'time' in got.columns:
            got['time'] = got['time'].astype('int64')

    print("")
    print("")
    print("Pyarrow:")
    print(expect.dtypes)
    print(expect)
    print("")
    print("cuDF:")
    print(got.dtypes)
    print(got)
    print("")

    #assert_eq(expect, got, check_categorical=False)

    # For debug testing
    #if 'time' in got.columns:
        #np.testing.assert_array_equal(expect['time'], got['time'])
    if 'date' in got.columns:
        np.testing.assert_array_equal(expect['date'], got['date'])
