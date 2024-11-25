# Copyright (c) 2024, NVIDIA CORPORATION.

import gc

import pyarrow as pa

import pylibcudf as plc


def test_gc_with_table_and_column_input_metadata(monkeypatch):
    class A(plc.io.types.TableInputMetadata):
        def __del__(self):
            print("Deleting A...")

    pa_table = pa.table(
        {"a": pa.array([1, 2, 3]), "b": pa.array(["a", "b", "c"])}
    )
    plc_table = plc.interop.from_arrow(pa_table)

    tbl_meta = A(plc_table)

    gc.disable()
    gc.collect()

    del tbl_meta

    # Circular reference creates an additional uncollectable object
    assert gc.collect() == 4

    gc.enable()
