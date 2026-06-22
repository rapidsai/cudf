# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

import gc
import io
import weakref

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from utils import assert_table_and_meta_eq

import pylibcudf as plc


@pytest.fixture
def pa_table():
    return pa.table({"a": [1, 2, 3], "b": ["x", "y", "z"]})


@pytest.fixture
def expected(pa_table):
    return plc.Table.from_arrow(pa_table)


@pytest.fixture
def parquet_data(tmp_path):
    tbl1 = pa.Table.from_pydict({"a": [3, 1, 4], "b": [1, 5, 9]})
    tbl2 = pa.Table.from_pydict({"a": [1, 6], "b": [1, 8]})

    path1 = tmp_path / "tbl1.parquet"
    path2 = tmp_path / "tbl2.parquet"

    pa.parquet.write_table(tbl1, path1)
    pa.parquet.write_table(tbl2, path2)

    return [path1, path2]


@pytest.fixture
def parquet_files(tmp_path, pa_table):
    paths = []
    for i in range(2):
        path = tmp_path / f"file_{i}.parquet"
        pq.write_table(pa_table, path)
        paths.append(path)
    return paths


def test_gc_with_table_and_column_input_metadata():
    class Foo(plc.io.types.TableInputMetadata):
        def __del__(self):
            pass

    plc_table = plc.Table.from_arrow(
        pa.table({"a": pa.array([1, 2, 3]), "b": pa.array(["a", "b", "c"])})
    )

    tbl_meta = Foo(plc_table)

    weak_tbl_meta = weakref.ref(tbl_meta)

    del tbl_meta

    gc.collect()

    assert weak_tbl_meta() is None


def test_num_rows_per_resource(parquet_data):
    source = plc.io.SourceInfo(parquet_data)
    options = plc.io.parquet.ParquetReaderOptions.builder(source).build()
    assert plc.io.parquet.read_parquet(options).num_rows_per_source == [3, 2]


def test_num_input_row_groups(parquet_data):
    source = plc.io.SourceInfo(parquet_data)
    options = plc.io.parquet.ParquetReaderOptions.builder(source).build()
    assert plc.io.parquet.read_parquet(options).num_input_row_groups == 2


def test_sourceinfo_from_paths(parquet_files, pa_table):
    source = plc.io.SourceInfo(parquet_files)
    opts = plc.io.parquet.ParquetReaderOptions.builder(source).build()
    got = plc.io.parquet.read_parquet(opts)
    expect = pa.concat_tables([pa_table] * len(parquet_files))
    assert_table_and_meta_eq(expect, got)


def test_sourceinfo_from_bytes(pa_table):
    buf = io.BytesIO()
    pq.write_table(pa_table, buf)
    source = plc.io.SourceInfo([buf.getvalue()])
    opts = plc.io.parquet.ParquetReaderOptions.builder(source).build()
    got = plc.io.parquet.read_parquet(opts)
    expect = pa_table
    assert_table_and_meta_eq(expect, got)


def test_sourceinfo_from_bytesio(pa_table):
    buf = io.BytesIO()
    pq.write_table(pa_table, buf)
    source = plc.io.SourceInfo([buf])
    opts = plc.io.parquet.ParquetReaderOptions.builder(source).build()
    got = plc.io.parquet.read_parquet(opts)
    expect = pa_table
    assert_table_and_meta_eq(expect, got)


def test_sourceinfo_with_empty_bytesio():
    # Test passing empty source
    plc.io.SourceInfo([io.BytesIO(), io.BytesIO()])


def test_sourceinfo_from_stringio(pa_table):
    expect = pa_table
    csv_data = "a,b\n1,x\n2,y\n3,z\n"

    source = plc.io.SourceInfo([io.StringIO(csv_data)])
    opts = plc.io.csv.CsvReaderOptions.builder(source).build()
    got = plc.io.csv.read_csv(opts)
    assert_table_and_meta_eq(expect, got)


def test_empty_bytes_buffer():
    empty_bytes = [b""]
    source = plc.io.SourceInfo(empty_bytes)

    options = plc.io.csv.CsvReaderOptions.builder(source).build()

    got = plc.io.csv.read_csv(options)

    expect = pa.table({})

    assert_table_and_meta_eq(expect, got)


# TODO: Test more IO types
