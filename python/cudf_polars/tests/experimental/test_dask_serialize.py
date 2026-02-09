# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest
from distributed.protocol import deserialize, serialize

import polars as pl
from polars.testing.asserts import assert_frame_equal

import rmm
from rmm.pylibrmm.stream import DEFAULT_STREAM

from cudf_polars.containers import DataFrame
from cudf_polars.experimental.dask_registers import register
from cudf_polars.utils.cuda_stream import get_dask_cuda_stream

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


@pytest.mark.filterwarnings(
    # If exceptions in threads aren't handled, they get raised as a warning by
    # Pytest. The warnings raised by this test correspond to unhandled
    # `ResourceWarning`s in `distributed.node`
    #
    # Since Pytest 8, these warnings get elevated to errors and exit the test
    # suite, so we selectively filter them here if the unraisable exception
    # concerns `socket.socket`
    "ignore:.*socket.socket.*:pytest.PytestUnraisableExceptionWarning"
)
@pytest.mark.parametrize(
    "polars_tbl",
    [
        pl.DataFrame(),
        pl.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6], "c": [7, 8, 9]}),
        pl.DataFrame({"a": [1, 2, 3]}),
        pl.DataFrame({"a": [1], "b": [2], "c": [3]}),
        pl.DataFrame({"a": ["a", "bb", "ccc"]}),
        pl.DataFrame({"a": [1, 2, None], "b": [None, 3, 4]}),
        pl.DataFrame({"a": range(int(1e7))}),
    ],
)
@pytest.mark.parametrize("protocol", ["cuda", "cuda_rmm", "dask"])
@pytest.mark.parametrize(
    "context",
    [
        None,
        {},
        {
            "stream": DEFAULT_STREAM,
            "device_mr": rmm.mr.get_current_device_resource(),
            "staging_device_buffer": rmm.DeviceBuffer(size=2**20),
        },
    ],
)
def test_dask_serialization_roundtrip(polars_tbl, protocol, context):
    stream = get_dask_cuda_stream()
    df = DataFrame.from_polars(polars_tbl, stream=stream)

    cuda_rmm = protocol == "cuda_rmm"
    protocol = "cuda" if protocol == "cuda_rmm" else protocol

    header, frames = serialize(
        df, on_error="raise", serializers=[protocol], context=context
    )
    if cuda_rmm:
        # Simulate Dask UCX transfers
        frames = [convert_to_rmm(f) for f in frames]
    res = deserialize(header, frames, deserializers=[protocol])

    assert_frame_equal(df.to_polars(), res.to_polars())

    # Check that we can serialize individual columns
    for column in df.columns:
        expect = DataFrame([column], stream=df.stream)

        header, frames = serialize(
            column, on_error="raise", serializers=[protocol], context=context
        )
        if cuda_rmm:
            # Simulate Dask UCX transfers
            frames = [convert_to_rmm(f) for f in frames]
        res = deserialize(header, frames, deserializers=[protocol])

        assert_frame_equal(
            expect.to_polars(), DataFrame([res], stream=df.stream).to_polars()
        )


def test_dask_serialization_error():
    df = DataFrame.from_polars(
        pl.DataFrame({"a": [1, 2, 3]}), stream=get_dask_cuda_stream()
    )

    header, frames = serialize(
        df,
        on_error="message",
        serializers=["dask"],
        context={
            "device_mr": rmm.mr.get_current_device_resource(),
            "staging_device_buffer": rmm.DeviceBuffer(size=2**20),
        },
    )
    assert header == {"serializer": "error"}
    assert "ValueError: " in str(frames)

    header, frames = serialize(
        df,
        on_error="message",
        serializers=["dask"],
        context={
            "stream": df.stream,
            "staging_device_buffer": rmm.DeviceBuffer(size=2**20),
        },
    )
    assert header == {"serializer": "error"}
    assert "ValueError: " in str(frames)
