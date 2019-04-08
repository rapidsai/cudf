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
    'orc_file',
    [
        'TestOrcFile.emptyFile.orc',
        'TestOrcFile.test1.orc'
    ]
)
@pytest.mark.parametrize('columns', [['boolean1', 'byte1', 'short1',
                                      'int1', 'long1', 'float1', 'double1']])
def test_orc_reader(datadir, orc_file, engine, columns):
    path = datadir / orc_file
    orcfile = pa.orc.ORCFile(path)

    expect = orcfile.read(columns=columns).to_pandas()
    got = cudf.read_orc(path, engine=engine, columns=columns)

    # cuDF's default currently handles bools differently
    # For bool, cuDF doesn't support it so convert it to int8
    if engine == 'cudf':
        if 'boolean1' in expect.columns:
            expect['boolean1'] = expect['boolean1'].astype('int8')

    print(expect)
    print(got)

    assert_eq(expect, got, check_categorical=False)
