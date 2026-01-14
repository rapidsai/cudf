# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Dask function registrations such as serializers and dispatch implementations."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar, overload

from dask.sizeof import sizeof as sizeof_dispatch
from dask.tokenize import normalize_token
from distributed.protocol import dask_deserialize, dask_serialize
from distributed.protocol.cuda import cuda_deserialize, cuda_serialize
from distributed.utils import log_errors

import pylibcudf as plc
import rmm

from cudf_polars.containers import Column, DataFrame, DataType
from cudf_polars.dsl.expressions.base import NamedExpr
from cudf_polars.utils.cuda_stream import get_dask_cuda_stream

if TYPE_CHECKING:
    from collections.abc import Hashable, Mapping

    from distributed import Client

    from rmm.pylibrmm.memory_resource import DeviceMemoryResource
    from rmm.pylibrmm.stream import Stream

    from cudf_polars.typing import ColumnHeader, ColumnOptions, DataFrameHeader


__all__ = ["DaskRegisterManager", "register"]


class DaskRegisterManager:  # pragma: no cover; Only used with Distributed cluster
    """Manager to ensure ensure serializer is only registered once."""

    _registered: bool = False
    _client_run_executed: ClassVar[set[str]] = set()

    @classmethod
    def register_once(cls) -> None:
        """Register Dask/cudf-polars serializers in calling process."""
        if not cls._registered:
            from cudf_polars.experimental.dask_registers import register

            register()
            cls._registered = True

    @classmethod
    def run_on_cluster(cls, client: Client) -> None:
        """Run register on the workers and scheduler once."""
        if client.id not in cls._client_run_executed:
            client.run(cls.register_once)
            client.run_on_scheduler(cls.register_once)
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
    ) -> tuple[
        DataFrameHeader | ColumnHeader, list[memoryview[bytes] | plc.gpumemoryview]
    ]:
        with log_errors():
            header, frames = x.serialize(stream=get_dask_cuda_stream())
            # Dask expect a list of frames
            return header, list(frames)

    @cuda_deserialize.register(DataFrame)
    def _(
        header: DataFrameHeader, frames: tuple[memoryview[bytes], plc.gpumemoryview]
    ) -> DataFrame:
        with log_errors():
            metadata, gpudata = frames  # TODO: check if this is a length-2 list...
            return DataFrame.deserialize(
                header,
                (metadata, plc.gpumemoryview(gpudata)),
                stream=get_dask_cuda_stream(),
            )

    @cuda_deserialize.register(Column)
    def _(
        header: ColumnHeader, frames: tuple[memoryview[bytes], plc.gpumemoryview]
    ) -> Column:
        with log_errors():
            metadata, gpudata = frames
            return Column.deserialize(
                header,
                (metadata, plc.gpumemoryview(gpudata)),
                stream=get_dask_cuda_stream(),
            )

    @overload
    def dask_serialize_column_or_frame(
        x: DataFrame,
    ) -> tuple[DataFrameHeader, tuple[memoryview[bytes], memoryview[bytes]]]: ...

    @overload
    def dask_serialize_column_or_frame(
        x: Column,
    ) -> tuple[ColumnHeader, tuple[memoryview[bytes], memoryview[bytes]]]: ...

    @dask_serialize.register(Column)
    def dask_serialize_column_or_frame(
        x: DataFrame | Column,
    ) -> tuple[
        DataFrameHeader | ColumnHeader, tuple[memoryview[bytes], memoryview[bytes]]
    ]:
        stream = get_dask_cuda_stream()
        with log_errors():
            header, (metadata, gpudata) = x.serialize(stream=stream)

            # For robustness, we check that the gpu data is contiguous
            cai = gpudata.__cuda_array_interface__
            assert len(cai["shape"]) == 1
            assert cai["strides"] is None or cai["strides"] == (1,)
            assert cai["typestr"] == "|u1"
            nbytes = cai["shape"][0]

            # Copy the gpudata to host memory
            gpudata_on_host: memoryview[bytes] = memoryview(
                rmm.DeviceBuffer(ptr=gpudata.ptr, size=nbytes).copy_to_host()
            )
            return header, (metadata, gpudata_on_host)

    @dask_deserialize.register(Column)
    def _(header: ColumnHeader, frames: tuple[memoryview[bytes], memoryview]) -> Column:
        with log_errors():
            assert len(frames) == 2
            # Copy the second frame (the gpudata in host memory) back to the gpu
            new_frames = (
                frames[0],
                plc.gpumemoryview(rmm.DeviceBuffer.to_device(frames[1])),
            )
            return Column.deserialize(header, new_frames, stream=get_dask_cuda_stream())

    @dask_serialize.register(DataFrame)
    def _(
        x: DataFrame, context: Mapping[str, Any] | None = None
    ) -> tuple[DataFrameHeader, tuple[memoryview[bytes], memoryview[bytes]]]:
        # Do regular serialization if no staging buffer is provided.
        if context is None or "staging_device_buffer" not in context:
            return dask_serialize_column_or_frame(x)

        # If a staging buffer is provided, we use `ChunkedPack` to
        # serialize the dataframe using the provided staging buffer.
        with log_errors():
            # Keyword arguments for `Column.__init__`.
            columns_kwargs: list[ColumnOptions] = [
                col.serialize_ctor_kwargs() for col in x.columns
            ]
            header: DataFrameHeader = {
                "columns_kwargs": columns_kwargs,
                "frame_count": 2,
            }
            if "stream" not in context:
                raise ValueError(
                    "context: stream must be given when staging_device_buffer is"
                )
            if "device_mr" not in context:
                raise ValueError(
                    "context: device_mr must be given when staging_device_buffer is"
                )
            stream: Stream = context["stream"]
            device_mr: DeviceMemoryResource = context["device_mr"]
            buf: rmm.DeviceBuffer = context["staging_device_buffer"]
            frame = plc.contiguous_split.ChunkedPack.create(
                x.table, buf.nbytes, stream, device_mr
            ).pack_to_host(buf)
            return header, frame

    @dask_deserialize.register(DataFrame)
    def _(
        header: DataFrameHeader, frames: tuple[memoryview[bytes], memoryview]
    ) -> DataFrame:
        with log_errors():
            assert len(frames) == 2
            # Copy the second frame (the gpudata in host memory) back to the gpu
            new_frames = (
                frames[0],
                plc.gpumemoryview(rmm.DeviceBuffer.to_device(frames[1])),
            )
            return DataFrame.deserialize(
                header, new_frames, stream=get_dask_cuda_stream()
            )

    @sizeof_dispatch.register(Column)
    def _(x: Column) -> int:
        """The total size of the device buffers used by the DataFrame or Column."""
        return x.obj.device_buffer_size()

    @sizeof_dispatch.register(DataFrame)
    def _(x: DataFrame) -> int:
        """The total size of the device buffers used by the DataFrame or Column."""
        return sum(c.obj.device_buffer_size() for c in x.columns)

    # Register rapidsmpf serializer if it's installed.
    try:
        from rapidsmpf.integrations.dask.spilling import register_dask_serialize

        register_dask_serialize()  # pragma: no cover; rapidsmpf dependency not included yet
    except ImportError:
        pass

    # Register the tokenizer for NamedExpr and DataType. This is a performance
    # optimization that speeds up tokenization for the most common types seen in
    # the Dask task graph.
    @normalize_token.register(NamedExpr)
    @normalize_token.register(DataType)
    def _(x: NamedExpr | DataType) -> Hashable:
        return hash(x)
