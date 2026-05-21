# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor

import pytest
from rapidsmpf.streaming.core.message import Message
from rapidsmpf.streaming.cudf.channel_metadata import (
    ChannelMetadata,
    OrderKey,
    OrderScheme,
    Partitioning,
)
from rapidsmpf.streaming.cudf.table_chunk import TableChunk

import polars as pl

import pylibcudf as plc

from cudf_polars.containers import DataFrame, DataType
from cudf_polars.dsl.ir import Empty, IRExecutionContext
from cudf_polars.streaming.actor_graph.collectives.common import reserve_op_id
from cudf_polars.streaming.actor_graph.collectives.orderscheme import (
    adjust_orderscheme,
)
from cudf_polars.streaming.actor_graph.utils import (
    gather_in_task_group,
    recv_metadata,
    send_metadata,
)

_SCHEMA = {"key": DataType(pl.Int32()), "val": DataType(pl.Int32())}
_NAMES = list(_SCHEMA)
_DTYPES = list(_SCHEMA.values())


def _make_scheme(
    context,
    boundary: int | tuple[int, ...],
    *,
    key_indices: tuple[int, ...] = (0,),
    strict: bool = True,
    stream,
) -> OrderScheme:
    boundary_values = (
        boundary if isinstance(boundary, tuple) else (boundary,) * len(key_indices)
    )
    boundary_df = DataFrame.from_polars(
        pl.DataFrame(
            {
                f"k{i}": pl.Series([value], dtype=pl.Int32())
                for i, value in enumerate(boundary_values)
            }
        ),
        stream,
    )
    return OrderScheme(
        [
            OrderKey(index, plc.types.Order.ASCENDING, plc.types.NullOrder.BEFORE)
            for index in key_indices
        ],
        TableChunk.from_pylibcudf_table(
            boundary_df.table,
            stream,
            exclusive_view=False,
            br=context.br(),
        ),
        strict_boundaries=strict,
    )


def _ref_ir() -> Empty:
    return Empty(_SCHEMA)


def _local_count(comm, scheme: OrderScheme) -> int:
    npartitions = scheme.num_boundaries + 1
    return sum(
        pid * comm.nranks // npartitions == comm.rank for pid in range(npartitions)
    )


def _frame(values: list[int]) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "key": pl.Series(values, dtype=pl.Int32()),
            "val": pl.Series(values, dtype=pl.Int32()),
        }
    )


def _chunk_to_polars(chunk: TableChunk) -> pl.DataFrame:
    return DataFrame.from_table(
        chunk.table_view(),
        _NAMES,
        _DTYPES,
        chunk.stream,
    ).to_polars()


async def _adjust_and_collect(
    context,
    comm,
    input_df: pl.DataFrame,
    input_scheme: OrderScheme,
    output_scheme: OrderScheme,
    *,
    collective_id: int | None = None,
) -> dict[int, pl.DataFrame]:
    ch_in = context.create_channel()
    ch_out = context.create_channel()
    stream = context.get_stream_from_pool()
    output: dict[int, pl.DataFrame] = {}

    async def _produce() -> None:
        df = DataFrame.from_polars(input_df, stream)
        await send_metadata(
            ch_out,
            context,
            ChannelMetadata(
                local_count=_local_count(comm, output_scheme),
                partitioning=Partitioning(output_scheme, "inherit"),
            ),
        )
        await ch_in.send(
            context,
            Message(
                comm.rank,
                TableChunk.from_pylibcudf_table(
                    df.table,
                    stream,
                    exclusive_view=True,
                    br=context.br(),
                ),
            ),
        )
        await ch_in.drain(context)

    async def _consume() -> None:
        await recv_metadata(ch_out, context)
        while (msg := await ch_out.recv(context)) is not None:
            output[msg.sequence_number] = _chunk_to_polars(
                TableChunk.from_message(msg, br=context.br())
            )

    with ThreadPoolExecutor(max_workers=1) as executor:
        ir_context = IRExecutionContext(
            executor, get_cuda_stream=context.get_stream_from_pool
        )
        await gather_in_task_group(
            _produce(),
            adjust_orderscheme(
                context,
                comm,
                _ref_ir(),
                ir_context,
                ch_out,
                ch_in,
                input_scheme,
                output_scheme,
                collective_id=collective_id,
            ),
            _consume(),
        )

    return output


async def _adjust_direct(
    context,
    comm,
    input_scheme: OrderScheme,
    output_scheme: OrderScheme,
    *,
    collective_id: int | None = None,
) -> None:
    ch_in = context.create_channel()
    ch_out = context.create_channel()
    with ThreadPoolExecutor(max_workers=1) as executor:
        ir_context = IRExecutionContext(
            executor, get_cuda_stream=context.get_stream_from_pool
        )
        await adjust_orderscheme(
            context,
            comm,
            _ref_ir(),
            ir_context,
            ch_out,
            ch_in,
            input_scheme,
            output_scheme,
            collective_id=collective_id,
        )


@pytest.mark.spmd
@pytest.mark.parametrize(
    "input_keys,output_keys,strict,error,match",
    [
        ((1,), (0,), True, NotImplementedError, "prefix"),
        ((0,), (0,), False, ValueError, "strict output"),
    ],
)
def test_adjust_orderscheme_rejects_invalid_schemes(
    spmd_engine, input_keys, output_keys, strict, error, match
) -> None:
    context = spmd_engine.context
    stream = context.get_stream_from_pool()
    input_scheme = _make_scheme(context, 4, key_indices=input_keys, stream=stream)
    output_scheme = _make_scheme(
        context,
        4,
        key_indices=output_keys,
        strict=strict,
        stream=stream,
    )

    with pytest.raises(error, match=match):
        asyncio.run(
            _adjust_direct(context, spmd_engine.comm, input_scheme, output_scheme)
        )


@pytest.mark.spmd
def test_adjust_orderscheme_requires_collective_id(spmd_engine) -> None:
    context = spmd_engine.context
    comm = spmd_engine.comm
    if comm.nranks == 1:
        pytest.skip("collective_id is only required for multi-rank runs.")

    stream = context.get_stream_from_pool()
    input_scheme = _make_scheme(context, 4, stream=stream)
    output_scheme = _make_scheme(context, 4, stream=stream)

    with pytest.raises(ValueError, match="collective_id"):
        asyncio.run(_adjust_direct(context, comm, input_scheme, output_scheme))


@pytest.mark.spmd
@pytest.mark.parametrize(
    "target_boundary,expected",
    [
        (3, {0: [0, 1, 2], 1: [3, 4, 5, 6, 7]}),
        (5, {0: [0, 1, 2, 3, 4], 1: [5, 6, 7]}),
    ],
)
def test_adjust_orderscheme_sparse_boundary_shift(
    spmd_engine, target_boundary, expected
) -> None:
    context = spmd_engine.context
    comm = spmd_engine.comm
    if comm.nranks != 2:
        pytest.skip("This test expects exactly two ranks.")

    keys = list(range(4)) if comm.rank == 0 else list(range(4, 8))
    stream = context.get_stream_from_pool()
    # Input sorted on (key, val) is also sorted on the target key prefix.
    input_scheme = _make_scheme(context, (4, 4), key_indices=(0, 1), stream=stream)
    output_scheme = _make_scheme(context, target_boundary, stream=stream)

    with reserve_op_id() as op_id:
        output = asyncio.run(
            _adjust_and_collect(
                context,
                comm,
                _frame(keys),
                input_scheme,
                output_scheme,
                collective_id=op_id,
            )
        )

    result = pl.concat(output.values())
    assert result["key"].to_list() == expected[comm.rank]
    assert result["val"].to_list() == expected[comm.rank]


@pytest.mark.spmd
@pytest.mark.parametrize(
    "target_boundary,expected",
    [
        (3, {0: [0, 1, 2], 1: [3, 4, 5, 6, 7]}),
        (0, {1: list(range(8))}),
    ],
)
def test_adjust_orderscheme_single_rank_no_collective(
    spmd_engine, target_boundary, expected
) -> None:
    context = spmd_engine.context
    comm = spmd_engine.comm
    if comm.nranks != 1:
        pytest.skip("This test covers the single-rank path.")

    stream = context.get_stream_from_pool()
    input_scheme = _make_scheme(context, 4, stream=stream)
    output_scheme = _make_scheme(context, target_boundary, stream=stream)
    output_by_pid = asyncio.run(
        _adjust_and_collect(
            context,
            comm,
            _frame(list(range(8))),
            input_scheme,
            output_scheme,
        )
    )

    assert set(output_by_pid) == set(expected)
    for pid, keys in expected.items():
        assert output_by_pid[pid]["key"].to_list() == keys
        assert output_by_pid[pid]["val"].to_list() == keys
