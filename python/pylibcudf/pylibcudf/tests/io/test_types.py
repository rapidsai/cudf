# Copyright (c) 2024-2025, NVIDIA CORPORATION.

import gc
import weakref

import pyarrow as pa
import pytest

import pylibcudf as plc


@pytest.fixture
def parquet_data(tmp_path):
    tbl1 = pa.Table.from_pydict({"a": [3, 1, 4], "b": [1, 5, 9]})
    tbl2 = pa.Table.from_pydict({"a": [1, 6], "b": [1, 8]})

    path1 = tmp_path / "tbl1.parquet"
    path2 = tmp_path / "tbl2.parquet"

    pa.parquet.write_table(tbl1, path1)
    pa.parquet.write_table(tbl2, path2)

    return [path1, path2]


def test_gc_with_table_and_column_input_metadata():
    class Foo(plc.io.types.TableInputMetadata):
        def __del__(self):
            pass

    pa_table = pa.table(
        {"a": pa.array([1, 2, 3]), "b": pa.array(["a", "b", "c"])}
    )
    plc_table = plc.interop.from_arrow(pa_table)

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


# TODO: Test more IO types
