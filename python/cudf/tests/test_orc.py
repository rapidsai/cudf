# Copyright (c) 2018, NVIDIA CORPORATION.

import cudf

import pytest
import pyarrow as pa


@pytest.mark.filterwarnings("ignore:Using CPU")
@pytest.mark.filterwarnings("ignore:Strings are not yet supported")
@pytest.mark.parametrize(
    'orc_file',
    [
        'TestOrcFile.emptyFile.orc',
        'TestOrcFile.test1.orc',
        'TestOrcFile.testDate1900.orc'
    ]
)
def test_orc_reader(orc_file):
    columns = ['boolean1', 'byte1', 'short1', 'int1', 'long1', 'float1',
               'double1']

    orcfile = pa.orc.ORCFile(orc_file)
    expect = orcfile.read(columns=columns)
    got = cudf.read_orc(orc_file, columns=columns)\
              .to_arrow(preserve_index=False)\
              .replace_schema_metadata()

    assert pa.Table.equals(expect, got)

    for column in columns:
        expect = orcfile.read(columns=[column])
        got = cudf.read_orc(orc_file, columns=[column])\
                  .to_arrow(preserve_index=False)\
                  .replace_schema_metadata()

    assert pa.Table.equals(expect, got)
