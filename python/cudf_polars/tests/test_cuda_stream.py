# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import rmm.pylibrmm.stream

import cudf_polars.utils.cuda_stream


def test_join_cuda_streams_single():
    # Test the optimization that a single stream returns itself.
    stream = rmm.pylibrmm.stream.Stream()
    joined = cudf_polars.utils.cuda_stream.get_joined_cuda_stream(
        rmm.pylibrmm.stream.Stream,
        upstreams=(stream,),
    )
    assert joined is stream

    # but multiple streams returns a new one.
    joined = cudf_polars.utils.cuda_stream.get_joined_cuda_stream(
        rmm.pylibrmm.stream.Stream,
        upstreams=(stream, stream),
    )
    assert joined is not stream
