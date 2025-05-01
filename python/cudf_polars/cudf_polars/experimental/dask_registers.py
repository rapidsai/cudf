# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Dask function registrations such as serializers and dispatch implementations."""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar, overload

from dask.sizeof import sizeof as sizeof_dispatch
from distributed.protocol import dask_deserialize, dask_serialize
from distributed.protocol.cuda import cuda_deserialize, cuda_serialize
from distributed.utils import log_errors

import pylibcudf as plc
import rmm

from cudf_polars.containers import Column, DataFrame

if TYPE_CHECKING:
    from distributed import Client

    from cudf_polars.typing import ColumnHeader, DataFrameHeader

__all__ = ["SerializerManager", "register"]


class SerializerManager:  # pragma: no cover; Only used with Distributed scheduler
    """Manager to ensure ensure serializer is only registered once."""

    _serializer_registered: bool = False
    _client_run_executed: ClassVar[set[str]] = set()

    @classmethod
    def register_serialize(cls) -> None:
        """Register Dask/cudf-polars serializers in calling process."""
        if not cls._serializer_registered:
            from cudf_polars.experimental.dask_registers import register

            register()
            cls._serializer_registered = True

    @classmethod
    def run_on_cluster(cls, client: Client) -> None:
        """Run serializer registration on the workers and scheduler."""
        if client.id not in cls._client_run_executed:
            client.run(cls.register_serialize)
            client.run_on_scheduler(cls.register_serialize)
            cls._client_run_executed.add(client.id)


def register() -> None:
    """Register dask serialization and dispatch functions."""

    @overload
    def serialize_column_or_frame(
        x: DataFrame,
    ) -> tuple[DataFrameHeader, list[memoryview]]: ...

    @overload
    def serialize_column_or_frame(
        x: Column,
    ) -> tuple[ColumnHeader, list[memoryview]]: ...

    @cuda_serialize.register((Column, DataFrame))
    def serialize_column_or_frame(
        x: DataFrame | Column,
    ) -> tuple[DataFrameHeader | ColumnHeader, list[memoryview]]:
        with log_errors():
            header, frames = x.serialize()
            return header, list(frames)  # Dask expect a list of frames

    @cuda_deserialize.register(DataFrame)
    def _(
        header: DataFrameHeader, frames: tuple[memoryview, plc.gpumemoryview]
    ) -> DataFrame:
        with log_errors():
            metadata, gpudata = frames  # TODO: check if this is a length-2 list...
            return DataFrame.deserialize(header, (metadata, plc.gpumemoryview(gpudata)))

    @cuda_deserialize.register(Column)
    def _(header: ColumnHeader, frames: tuple[memoryview, plc.gpumemoryview]) -> Column:
        with log_errors():
            metadata, gpudata = frames
            return Column.deserialize(header, (metadata, plc.gpumemoryview(gpudata)))

    @overload
    def dask_serialize_column_or_frame(
        x: DataFrame,
    ) -> tuple[DataFrameHeader, tuple[memoryview, memoryview]]: ...

    @overload
    def dask_serialize_column_or_frame(
        x: Column,
    ) -> tuple[ColumnHeader, tuple[memoryview, memoryview]]: ...

    @dask_serialize.register((Column, DataFrame))
    def dask_serialize_column_or_frame(
        x: DataFrame | Column,
    ) -> tuple[DataFrameHeader | ColumnHeader, tuple[memoryview, memoryview]]:
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
    def _(header: DataFrameHeader, frames: tuple[memoryview, memoryview]) -> DataFrame:
        with log_errors():
            assert len(frames) == 2
            # Copy the second frame (the gpudata in host memory) back to the gpu
            frames = frames[0], plc.gpumemoryview(rmm.DeviceBuffer.to_device(frames[1]))
            return DataFrame.deserialize(header, frames)

    @dask_deserialize.register(Column)
    def _(header: ColumnHeader, frames: tuple[memoryview, memoryview]) -> Column:
        with log_errors():
            assert len(frames) == 2
            # Copy the second frame (the gpudata in host memory) back to the gpu
            frames = frames[0], plc.gpumemoryview(rmm.DeviceBuffer.to_device(frames[1]))
            return Column.deserialize(header, frames)

    @sizeof_dispatch.register(Column)
    def _(x: Column) -> int:
        ret = 0
        if x.obj.data() is not None:
            ret += x.obj.data().nbytes
        if x.obj.null_mask() is not None:
            ret += x.obj.null_mask().nbytes
        if x.obj.children() is not None:
            ret += sum(sizeof_dispatch(c) for c in x.obj.children())
        return ret

    @sizeof_dispatch.register(DataFrame)
    def _(x: DataFrame) -> int:
        return sum(sizeof_dispatch(c) for c in x.columns)
