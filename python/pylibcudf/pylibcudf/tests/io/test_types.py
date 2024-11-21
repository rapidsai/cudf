# Copyright (c) 2024, NVIDIA CORPORATION.

import gc
import io
from contextlib import redirect_stdout

import pyarrow as pa

import pylibcudf as plc


def test_gc_with_table_and_column_input_metadata():
    buffer = io.StringIO()

    with redirect_stdout(buffer):
        pa_table = pa.table(
            {"a": pa.array([1, 2, 3]), "b": pa.array(["a", "b", "c"])}
        )

        plc_table = plc.interop.from_arrow(pa_table)

        tbl_meta = plc.io.types.TableInputMetadata(plc_table)

        del tbl_meta

        collected = gc.collect()  # force gc

    output = buffer.getvalue()

    # the circular reference creates one uncollectable object, 2+1+1 = 4
    assert (
        collected == 4
    ), f"Expected 4 collected objects, but got {collected} objects"
    assert (
        output.count("ColumnInMetadata") == 2
    ), f"Expected 2 deleted column objects, but got {output.count("ColumnInMetadata")} objects"
    assert (
        output.count("TableInputMetadata") == 1
    ), f"Expected 1 deleted table object, but got {output.count("TableInputMetadata")} objects"
