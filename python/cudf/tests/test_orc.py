# Copyright (c) 2018, NVIDIA CORPORATION.

import cudf

import pytest
import pyarrow as pa


@pytest.fixture(scope='module')
def datadir(datadir):
    return datadir / 'orc'


@pytest.mark.filterwarnings("ignore:Using CPU")
@pytest.mark.filterwarnings("ignore:Strings are not yet supported")
@pytest.mark.parametrize(
    'orc_file',
    [
        'TestOrcFile.emptyFile.orc',
        'TestOrcFile.test1.orc'
    ]
)
def test_orc_reader(datadir, orc_file):
    columns = ['boolean1', 'byte1', 'short1', 'int1', 'long1', 'float1',
               'double1']

    path = datadir / orc_file

    orcfile = pa.orc.ORCFile(path)
    expect = orcfile.read(columns=columns)
    got = cudf.read_orc(path, columns=columns)\
              .to_arrow(preserve_index=False)\
              .replace_schema_metadata()

    assert pa.Table.equals(expect, got)

    for column in columns:
        expect = orcfile.read(columns=[column])
        got = cudf.read_orc(path, columns=[column])\
                  .to_arrow(preserve_index=False)\
                  .replace_schema_metadata()

    assert pa.Table.equals(expect, got)
