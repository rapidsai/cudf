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
    def _(x):
        with log_errors():
            return x.serialize()

    @cuda_deserialize.register(DataFrame)
    def _(header, frames):
        with log_errors():
            return DataFrame.deserialize(header, frames)

    @dask_serialize.register(DataFrame)
    def _(x):
        with log_errors():
            header, frames = x.serialize()
            # Copy GPU buffers to host and record it in the header
            gpu_frames = [
                i
                for i in range(len(frames))
                if isinstance(frames[i], plc.gpumemoryview)
            ]
            for i in gpu_frames:
                frames[i] = memoryview(cupy.asnumpy(frames[i]))
            header["gpu_frames"] = gpu_frames
            return header, frames

    @dask_deserialize.register(DataFrame)
    def _(header, frames):
        with log_errors():
            # Copy GPU buffers back to device memory
            for i in header.pop("gpu_frames"):
                frames[i] = plc.gpumemoryview(cupy.asarray(frames[i]))
            return DataFrame.deserialize(header, frames)

except ImportError:
    pass  # distributed is probably not installed on the system
