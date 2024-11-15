# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pyarrow as pa
import pytest
from distributed.protocol import deserialize, serialize

from polars.testing.asserts import assert_frame_equal

import pylibcudf as plc

from cudf_polars.containers import DataFrame
from cudf_polars.experimental.dask_serialize import register

# Must register serializers before running tests
register()


@pytest.mark.parametrize(
    "arrow_tbl",
    [
        pa.table([]),
        pa.table({"a": [1, 2, 3], "b": [4, 5, 6], "c": [7, 8, 9]}),
        pa.table({"a": [1, 2, 3]}),
        pa.table({"a": [1], "b": [2], "c": [3]}),
        pa.table({"a": ["a", "bb", "ccc"]}),
        pa.table({"a": [1, 2, None], "b": [None, 3, 4]}),
    ],
)
@pytest.mark.parametrize("protocol", ["cuda", "dask"])
def test_dask_serialization_roundtrip(arrow_tbl, protocol):
    plc_tbl = plc.interop.from_arrow(arrow_tbl)
    df = DataFrame.from_table(plc_tbl, names=arrow_tbl.column_names)

    header, frames = serialize(df, on_error="raise", serializers=[protocol])
    res = deserialize(header, frames, deserializers=[protocol])

    assert_frame_equal(df.to_polars(), res.to_polars())
