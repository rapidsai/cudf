# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Dask serialization."""

from __future__ import annotations

from distributed.protocol import dask_deserialize, dask_serialize
from distributed.protocol.cuda import cuda_deserialize, cuda_serialize
from distributed.utils import log_errors

import pylibcudf as plc
import rmm

from cudf_polars.containers import DataFrame

__all__ = ["register"]


def register() -> None:
    """Register dask serialization routines for DataFrames."""

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
            header, (metadata, gpudata) = x.serialize()

            # For robustness, we check that the gpu data is contiguous
            cai = gpudata.__cuda_array_interface__
            assert len(cai["shape"]) == 1
            assert cai["strides"] is None or cai["strides"] == (1,)
            assert cai["typestr"] == "|u1"
            nbytes = cai["shape"][0]

            # Copy the gpudata to host memory
            gpudata_on_host = memoryview(
                rmm.DeviceBuffer(ptr=gpudata.ptr, size=nbytes).copy_to_host()
            )
            return header, (metadata, gpudata_on_host)

    @dask_deserialize.register(DataFrame)
    def _(header, frames) -> DataFrame:
        with log_errors():
            assert len(frames) == 2
            # Copy the second frame (the gpudata in host memory) back to the gpu
            frames = frames[0], plc.gpumemoryview(rmm.DeviceBuffer.to_device(frames[1]))
            return DataFrame.deserialize(header, frames)
