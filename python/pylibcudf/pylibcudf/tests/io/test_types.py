# Copyright (c) 2024, NVIDIA CORPORATION.

import gc
import weakref

import pyarrow as pa

import pylibcudf as plc


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
