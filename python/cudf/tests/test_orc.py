# Copyright (c) 2019, NVIDIA CORPORATION.

import cudf
from cudf.tests.utils import assert_eq

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
                                   'int1', 'long1', 'float1', 'double1']]
        #['TestOrcFile.testSnappy.orc', None],
    ]
)
def test_orc_reader(datadir, orc_args, engine):
    path = datadir / orc_args[0]
    orcfile = pa.orc.ORCFile(path)
    columns = orc_args[1]

    expect = orcfile.read(columns=columns).to_pandas()
    got = cudf.read_orc(path, engine=engine, columns=columns)

    # cuDF's default currently handles bools differently
    # For bool, cuDF doesn't support it so convert it to int8
    if engine == 'cudf':
        if 'boolean1' in expect.columns:
            expect['boolean1'] = expect['boolean1'].astype('int8')

    print("")
    print("")
    print("Pyarrow:")
    print(expect)
    print("")
    print("cuDF:")
    print(got)
    print("")

    assert_eq(expect, got, check_categorical=False)
