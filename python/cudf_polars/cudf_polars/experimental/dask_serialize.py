# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Dask serialization."""

from __future__ import annotations

import pylibcudf as plc

from cudf_polars.containers import DataFrame

try:
    import cupy
    from distributed.protocol import (
        dask_deserialize,
        dask_serialize,
    )
    from distributed.protocol.cuda import (
        cuda_deserialize,
        cuda_serialize,
    )
    from distributed.utils import log_errors

    @cuda_serialize.register(DataFrame)
    def _(x: DataFrame):
        with log_errors():
            header, frames = x.serialize()
            return header, list(frames)  # Dask expect a list of frames

    @cuda_deserialize.register(DataFrame)
    def _(header, frames):
        with log_errors():
            assert len(frames) == 2
            return DataFrame.deserialize(header, tuple(frames))

    @dask_serialize.register(DataFrame)
    def _(x: DataFrame):
        with log_errors():
            header, (metadata, gpudata_on_host) = x.serialize()
            return header, (metadata, memoryview(cupy.asnumpy(gpudata_on_host)))

    @dask_deserialize.register(DataFrame)
    def _(header, frames):
        with log_errors():
            assert len(frames) == 2
            # Copy the second frame (the gpudata in host memory), back to the gpu
            frames = frames[0], plc.gpumemoryview(cupy.asarray(frames[1]))
            return DataFrame.deserialize(header, frames)

except ImportError:
    pass  # distributed is probably not installed on the system
