# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pyarrow as pa
import pytest
from distributed.protocol import deserialize, serialize

from polars.testing.asserts import assert_frame_equal

import pylibcudf as plc
import rmm

from cudf_polars.containers import DataFrame
from cudf_polars.experimental.dask_serialize import register

# Must register serializers before running tests
register()


def convert_to_rmm(frame):
    """Convert frame to RMM to simulate Dask UCX transfers."""
    if hasattr(frame, "__cuda_array_interface__"):
        buf = rmm.DeviceBuffer(size=frame.nbytes)
        buf.copy_from_device(frame)
        return buf
    else:
        return frame


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
@pytest.mark.parametrize("protocol", ["cuda", "cuda_rmm", "dask"])
def test_dask_serialization_roundtrip(arrow_tbl, protocol):
    plc_tbl = plc.interop.from_arrow(arrow_tbl)
    df = DataFrame.from_table(plc_tbl, names=arrow_tbl.column_names)

    cuda_rmm = protocol == "cuda_rmm"
    protocol = "cuda" if protocol == "cuda_rmm" else protocol

    header, frames = serialize(df, on_error="raise", serializers=[protocol])
    if cuda_rmm:
        # Simulate Dask UCX transfers
        frames = [convert_to_rmm(f) for f in frames]
    res = deserialize(header, frames, deserializers=[protocol])

    assert_frame_equal(df.to_polars(), res.to_polars())

    # Check that we can serialize individual columns
    for column in df.columns:
        expect = DataFrame([column])

        header, frames = serialize(column, on_error="raise", serializers=[protocol])
        if cuda_rmm:
            # Simulate Dask UCX transfers
            frames = [convert_to_rmm(f) for f in frames]
        res = deserialize(header, frames, deserializers=[protocol])

        assert_frame_equal(expect.to_polars(), DataFrame([res]).to_polars())
