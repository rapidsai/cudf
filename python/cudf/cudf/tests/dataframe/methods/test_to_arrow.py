# Copyright (c) 2023-2025, NVIDIA CORPORATION.

import numpy as np
import pandas as pd
import pyarrow as pa
import pytest

import cudf
from cudf.testing import assert_eq


@pytest.mark.parametrize("preserve_index", [False, True, None])
def test_dataframe_to_arrow_preserve_index(preserve_index):
    df = cudf.DataFrame({"x": ["cat", "dog"] * 5})
    pf = df.to_pandas()
    expect = pa.Table.from_pandas(pf, preserve_index=preserve_index).schema
    got = df.to_arrow(preserve_index=preserve_index).schema
    assert expect == got


def test_dataframe_list_round_trip():
    data = [{"text": "hello", "list_col": np.asarray([1, 2], dtype="uint32")}]
    cudf_arrow = cudf.DataFrame(data).to_arrow()
    pdf_arrow = pa.Table.from_pandas(pd.DataFrame(data))

    for metadata in [
        None,
        pdf_arrow.schema.metadata,
        cudf_arrow.schema.metadata,
    ]:
        schema = pa.schema(
            [
                pa.field("text", pa.string()),
                pa.field("list_col", pa.list_(pa.uint32())),
            ],
            metadata=metadata,
        )

        data = {"text": ["asd", "pqr"], "list_col": [[1, 2, 3], [4, 5]]}

        table = pa.Table.from_pydict(data, schema=schema)
        assert_eq(table.to_pandas(), pd.DataFrame(data))
