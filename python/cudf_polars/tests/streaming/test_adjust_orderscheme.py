# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING

import pytest

import polars as pl

import pylibcudf as plc
from cudf_streaming.channel_metadata import (
    OrderKey,
    OrderScheme,
    Ordering,
)
from cudf_streaming.table_chunk import TableChunk
from rapidsmpf.streaming.core.message import Message

from cudf_polars.containers import DataFrame, DataType
from cudf_polars.dsl.ir import Empty, IRExecutionContext
from cudf_polars.streaming.actor_graph.collectives.common import reserve_op_id
from cudf_polars.streaming.actor_graph.collectives.orderscheme import (
    adjust_orderscheme,
)
from cudf_polars.streaming.actor_graph.utils import gather_in_task_group

if TYPE_CHECKING:
    from rapidsmpf.communicator.communicator import Communicator
    from rapidsmpf.streaming.core.context import Context
    from rmm.pylibrmm.stream import Stream

    from cudf_polars.engine.spmd import SPMDEngine

_SCHEMA = {"key": DataType(pl.Int32()), "val": DataType(pl.Int32())}
_NAMES = list(_SCHEMA)
_DTYPES = list(_SCHEMA.values())
_Boundary = int | tuple[int, ...]
_ExpectedPartitions = dict[int, list[int]]
_ExpectedByRank = dict[int, _ExpectedPartitions]


def _boundary_value(boundary: _Boundary, index: int) -> int:
    return boundary[index] if isinstance(boundary, tuple) else boundary


def _make_scheme(
    context: Context,
    boundary: _Boundary | list[_Boundary],
    *,
    key_indices: tuple[int, ...] = (0,),
    strict: bool = True,
    stream: Stream,
) -> OrderScheme:
    boundary_rows: list[_Boundary] = (
        boundary if isinstance(boundary, list) else [boundary]
    )
    boundary_df = DataFrame.from_polars(
        pl.DataFrame(
            {
                f"k{i}": pl.Series(
                    [_boundary_value(value, i) for value in boundary_rows],
                    dtype=pl.Int32(),
                )
                for i in range(len(key_indices))
            }
        ),
        stream,
    )
    return OrderScheme(
        [
            Ordering(
                [
                    OrderKey(
                        index,
                        plc.types.Order.ASCENDING,
                        plc.types.NullOrder.BEFORE,
                    )
                    for index in key_indices
                ],
                TableChunk.from_pylibcudf_table(
                    boundary_df.table,
                    stream,
                    exclusive_view=True,
                    br=context.br(),
                ),
                strict_boundaries=strict,
            )
        ]
    )


def _payload_value(key: int) -> int:
    return key * 10 + 1


def _frame(values: list[int]) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "key": pl.Series(values, dtype=pl.Int32()),
            "val": pl.Series([_payload_value(v) for v in values], dtype=pl.Int32()),
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
    context: Context,
    comm: Communicator,
    input_df: pl.DataFrame | list[pl.DataFrame],
    input_scheme: OrderScheme,
    output_scheme: OrderScheme,
    *,
    collective_id: int | None = None,
) -> dict[int, pl.DataFrame]:
    """Run adjustment and collect output chunks by partition ID."""
    ch_in = context.create_channel()
    ch_out = context.create_channel()
    stream = context.br().stream_pool.get_stream()
    output: dict[int, pl.DataFrame] = {}

    async def _produce() -> None:
        input_dfs = input_df if isinstance(input_df, list) else [input_df]
        for sequence_number, df in enumerate(input_dfs):
            cudf_df = DataFrame.from_polars(df, stream)
            await ch_in.send(
                context,
                Message(
                    sequence_number,
                    TableChunk.from_pylibcudf_table(
                        cudf_df.table,
                        stream,
                        exclusive_view=True,
                        br=context.br(),
                    ),
                ),
            )
        await ch_in.drain(context)

    async def _consume() -> None:
        while (msg := await ch_out.recv(context)) is not None:
            output[msg.sequence_number] = _chunk_to_polars(
                TableChunk.from_message(msg, br=context.br())
            )

    with ThreadPoolExecutor(max_workers=1) as executor:
        ir_context = IRExecutionContext(
            executor, get_cuda_stream=context.br().stream_pool.get_stream
        )
        await gather_in_task_group(
            _produce(),
            adjust_orderscheme(
                context,
                comm,
                Empty(_SCHEMA),
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


def _assert_partition_output(
    output: dict[int, pl.DataFrame], expected: dict[int, list[int]]
) -> None:
    """Assert output partition IDs and per-partition row order."""
    assert set(output) == set(expected)
    for pid, keys in expected.items():
        assert output[pid]["key"].to_list() == keys
        assert output[pid]["val"].to_list() == [_payload_value(key) for key in keys]


async def _adjust_direct(
    context: Context,
    comm: Communicator,
    input_scheme: OrderScheme,
    output_scheme: OrderScheme,
    *,
    collective_id: int | None = None,
) -> None:
    ch_in = context.create_channel()
    ch_out = context.create_channel()
    with ThreadPoolExecutor(max_workers=1) as executor:
        ir_context = IRExecutionContext(
            executor, get_cuda_stream=context.br().stream_pool.get_stream
        )
        await adjust_orderscheme(
            context,
            comm,
            Empty(_SCHEMA),
            ir_context,
            ch_out,
            ch_in,
            input_scheme,
            output_scheme,
            collective_id=collective_id,
        )


@pytest.mark.spmd
@pytest.mark.parametrize(
    "input_keys,output_keys,strict,err,match",
    [
        ((1,), (0,), True, NotImplementedError, "prefix"),
        ((0,), (0,), False, ValueError, "strict output"),
    ],
)
def test_adjust_orderscheme_rejects_invalid_schemes(
    spmd_engine: SPMDEngine,
    input_keys: tuple[int, ...],
    output_keys: tuple[int, ...],
    strict: bool,  # noqa: FBT001
    err: type[Exception],
    match: str,
) -> None:
    context = spmd_engine.context
    stream = context.br().stream_pool.get_stream()
    input_scheme = _make_scheme(context, 4, key_indices=input_keys, stream=stream)
    output_scheme = _make_scheme(
        context,
        4,
        key_indices=output_keys,
        strict=strict,
        stream=stream,
    )

    with pytest.raises(err, match=match):
        asyncio.run(
            _adjust_direct(context, spmd_engine.comm, input_scheme, output_scheme)
        )


@pytest.mark.spmd
def test_adjust_orderscheme_requires_collective_id(
    spmd_engine: SPMDEngine,
) -> None:
    context = spmd_engine.context
    comm = spmd_engine.comm
    if comm.nranks == 1:
        pytest.skip("collective_id is only required for multi-rank runs.")

    stream = context.br().stream_pool.get_stream()
    input_scheme = _make_scheme(context, 4, stream=stream)
    output_scheme = _make_scheme(context, 4, stream=stream)

    with pytest.raises(ValueError, match="collective_id"):
        asyncio.run(_adjust_direct(context, comm, input_scheme, output_scheme))


@pytest.mark.spmd
@pytest.mark.parametrize(
    "target_boundary,expected",
    [
        (3, {0: {0: [0, 1, 2]}, 1: {1: [3, 4, 5, 6, 7]}}),
        (5, {0: {0: [0, 1, 2, 3, 4]}, 1: {1: [5, 6, 7]}}),
    ],
)
def test_adjust_orderscheme_sparse_boundary_shift(
    spmd_engine: SPMDEngine,
    target_boundary: int,
    expected: _ExpectedByRank,
) -> None:
    context = spmd_engine.context
    comm = spmd_engine.comm
    if comm.nranks != 2:
        pytest.skip("This test expects exactly two ranks.")

    keys = list(range(4)) if comm.rank == 0 else list(range(4, 8))
    stream = context.br().stream_pool.get_stream()
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

    _assert_partition_output(output, expected[comm.rank])


@pytest.mark.spmd
def test_adjust_orderscheme_emits_empty_owned_partitions(
    spmd_engine: SPMDEngine,
) -> None:
    context = spmd_engine.context
    comm = spmd_engine.comm
    if comm.nranks != 2:
        pytest.skip("This test expects exactly two ranks.")

    keys = [0, 1, 2] if comm.rank == 0 else [5, 8]
    stream = context.br().stream_pool.get_stream()
    input_scheme = _make_scheme(context, 5, stream=stream)
    output_scheme = _make_scheme(context, [3, 5, 7], stream=stream)

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

    expected = {
        0: {0: [0, 1, 2], 1: []},
        1: {2: [5], 3: [8]},
    }[comm.rank]
    _assert_partition_output(output, expected)


@pytest.mark.spmd
def test_adjust_orderscheme_all_empty_input(spmd_engine: SPMDEngine) -> None:
    context = spmd_engine.context
    comm = spmd_engine.comm
    stream = context.br().stream_pool.get_stream()
    input_scheme = _make_scheme(context, 5, stream=stream)
    output_scheme = _make_scheme(context, [3, 5, 7], stream=stream)
    output_npartitions = output_scheme.orderings[0].num_boundaries + 1
    expected: _ExpectedPartitions = {
        pid: []
        for pid in range(output_npartitions)
        if pid * comm.nranks // output_npartitions == comm.rank
    }

    if comm.nranks == 1:
        output = asyncio.run(
            _adjust_and_collect(
                context,
                comm,
                _frame([]),
                input_scheme,
                output_scheme,
            )
        )
    else:
        with reserve_op_id() as op_id:
            output = asyncio.run(
                _adjust_and_collect(
                    context,
                    comm,
                    _frame([]),
                    input_scheme,
                    output_scheme,
                    collective_id=op_id,
                )
            )

    _assert_partition_output(output, expected)


@pytest.mark.spmd
@pytest.mark.parametrize(
    "target_boundary,expected",
    [
        (3, {0: [0, 1, 2], 1: [3, 4, 5, 6, 7]}),
        (0, {0: [], 1: list(range(8))}),
    ],
)
def test_adjust_orderscheme_single_rank_no_collective(
    spmd_engine: SPMDEngine,
    target_boundary: int,
    expected: _ExpectedPartitions,
) -> None:
    context = spmd_engine.context
    comm = spmd_engine.comm
    if comm.nranks != 1:
        pytest.skip("This test covers the single-rank path.")

    stream = context.br().stream_pool.get_stream()
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

    _assert_partition_output(output_by_pid, expected)


@pytest.mark.spmd
def test_adjust_orderscheme_multi_chunk_input(spmd_engine: SPMDEngine) -> None:
    context = spmd_engine.context
    comm = spmd_engine.comm
    if comm.nranks != 1:
        pytest.skip("This test covers local chunk accumulation.")

    stream = context.br().stream_pool.get_stream()
    input_scheme = _make_scheme(context, 4, stream=stream)
    output_scheme = _make_scheme(context, 4, stream=stream)
    output = asyncio.run(
        _adjust_and_collect(
            context,
            comm,
            [_frame([0, 1]), _frame([2, 3, 4, 5]), _frame([6, 7])],
            input_scheme,
            output_scheme,
        )
    )

    _assert_partition_output(output, {0: [0, 1, 2, 3], 1: [4, 5, 6, 7]})
