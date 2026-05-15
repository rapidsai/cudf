# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio

import pytest
from rapidsmpf.streaming.core.message import Message
from rapidsmpf.streaming.cudf.channel_metadata import OrderKey
from rapidsmpf.streaming.cudf.table_chunk import TableChunk

import polars as pl

import pylibcudf as plc

from cudf_polars.containers import DataFrame, DataType
from cudf_polars.dsl.ir import Empty, IRExecutionContext
from cudf_polars.experimental.rapidsmpf.collectives.common import reserve_op_id
from cudf_polars.experimental.rapidsmpf.collectives.sort import extract_orderscheme
from cudf_polars.experimental.rapidsmpf.frontend.options import StreamingOptions
from cudf_polars.experimental.rapidsmpf.utils import gather_in_task_group
from cudf_polars.testing.asserts import assert_gpu_result_equal


@pytest.fixture
def engine(streaming_engine_factory):
    return streaming_engine_factory(
        StreamingOptions(
            max_rows_per_partition=3,
            fallback_mode="raise",
            raise_on_fail=True,
        ),
    )


@pytest.fixture
def engine_large(streaming_engine_factory):
    return streaming_engine_factory(
        StreamingOptions(
            max_rows_per_partition=2_100,
            fallback_mode="raise",
            raise_on_fail=True,
        ),
    )


@pytest.fixture(scope="module")
def df():
    return pl.LazyFrame(
        {
            "x": [1, 2, 3, 4, 5, 6, 7],
            "y": [1, 6, 7, 2, 5, 4, 3],
            "z": ["e", "c", "b", "g", "a", "f", "d"],
        }
    )


def large_frames():
    x = [1.0] * 10_000
    x[-1] = float("nan")
    y = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] * 1000

    yield pytest.param(
        pl.LazyFrame(
            {
                "x": x,
            }
        ),
        ["x"],
        False,
        id="all_equal_one_nan",
    )

    yield pytest.param(
        pl.LazyFrame(
            {
                "x": x,
                "y": y,
            }
        ),
        ["x", "y"],
        False,
        id="two_cols",
    )

    idx = list(range(10_000))
    yield pytest.param(
        pl.LazyFrame(
            {
                "x": x,
                "y": y,
                "idx": idx,
            }
        ),
        ["x", "y"],
        True,
        id="two_col_stable",
    )


def test_sort(df, engine):
    q = df.sort(by=["y", "z"])
    assert_gpu_result_equal(q, engine=engine)


@pytest.mark.parametrize("large_df,by,stable", list(large_frames()))
@pytest.mark.parametrize(
    "nulls_last,descending", [(True, False), (True, True), (False, True)]
)
def test_large_sort(large_df, by, engine_large, stable, nulls_last, descending):
    q = large_df.sort(
        by=by, nulls_last=nulls_last, maintain_order=stable, descending=descending
    )
    assert_gpu_result_equal(q, engine=engine_large)


def test_sort_head(df, engine):
    q = df.sort(by=["y", "z"]).head(2)
    assert_gpu_result_equal(q, engine=engine)


def test_sort_tail(df, engine):
    q = df.sort(by=["y", "z"]).tail(2)
    assert_gpu_result_equal(q, engine=engine)


@pytest.mark.parametrize("offset", [1, -4])
def test_sort_slice(df, engine, offset):
    # Slice in the middle, which distributed sorts need to be careful with
    q = df.sort(by=["y", "z"]).slice(offset, 2)
    with pytest.raises(
        NotImplementedError,
        match="This slice not supported for multiple partitions.",
    ):
        assert_gpu_result_equal(q, engine=engine)


def test_sort_after_sparse_join(streaming_engine_factory):
    engine = streaming_engine_factory(
        StreamingOptions(max_rows_per_partition=4, raise_on_fail=True),
    )
    left = pl.LazyFrame({"foo": list(range(5)), "bar": list(range(5))})
    right = pl.LazyFrame({"foo": list(range(1))})
    q = left.join(right, on="foo", how="inner").sort(by=["foo"])
    assert_gpu_result_equal(q, engine=engine)


async def _send_sorted_chunks(
    context,
    ch,
    *,
    key_start: int,
    n_chunks: int,
    n_rows: int,
) -> None:
    stream = context.get_stream_from_pool()
    for i in range(n_chunks):
        start = key_start + i * n_rows
        tbl = DataFrame.from_polars(
            pl.DataFrame(
                {"key": pl.Series(range(start, start + n_rows), dtype=pl.Int32())}
            ),
            stream,
        ).table
        await ch.send(
            context,
            Message(
                i,
                TableChunk.from_pylibcudf_table(
                    tbl, stream, exclusive_view=True, br=context.br()
                ),
            ),
        )
    await ch.drain(context)


@pytest.mark.spmd
@pytest.mark.parametrize("n_chunks", [2, 4])
def test_extract_orderscheme(spmd_engine, n_chunks) -> None:
    context = spmd_engine.context
    comm = spmd_engine.comm

    n_rows = 4
    key_start = comm.rank * n_chunks * n_rows
    order_keys = [OrderKey(0, plc.types.Order.ASCENDING, plc.types.NullOrder.BEFORE)]
    schema_ir = Empty({"key": DataType(pl.Int32())})
    ir_context = IRExecutionContext(get_cuda_stream=context.get_stream_from_pool)

    async def _run():
        ch = context.create_channel()
        with reserve_op_id() as op_id:
            _, scheme = await gather_in_task_group(
                _send_sorted_chunks(
                    context, ch, key_start=key_start, n_chunks=n_chunks, n_rows=n_rows
                ),
                extract_orderscheme(
                    context, comm, schema_ir, ir_context, ch, order_keys, op_id
                ),
            )
        return scheme

    scheme = asyncio.run(_run())

    assert scheme is not None
    assert scheme.keys == tuple(order_keys)
    assert scheme.strict_boundaries  # all keys are distinct integers
    assert scheme.num_boundaries == comm.nranks * n_chunks - 1

    # Verify actual boundary values: start of each partition except the first
    expected_keys = [i * n_rows for i in range(1, comm.nranks * n_chunks)]
    tbl, bstream = scheme.get_boundaries()
    actual_keys = (
        DataFrame.from_table(tbl, ["key"], [DataType(pl.Int32())], stream=bstream)
        .to_polars()["key"]
        .to_list()
    )
    assert actual_keys == expected_keys


@pytest.mark.spmd
def test_extract_orderscheme_unsorted(spmd_engine) -> None:
    context = spmd_engine.context
    comm = spmd_engine.comm

    order_keys = [OrderKey(0, plc.types.Order.ASCENDING, plc.types.NullOrder.BEFORE)]
    schema_ir = Empty({"key": DataType(pl.Int32())})
    ir_context = IRExecutionContext(get_cuda_stream=context.get_stream_from_pool)

    async def _run():
        ch = context.create_channel()
        stream = context.get_stream_from_pool()

        async def _send() -> None:
            # Two locally-sorted chunks that are globally out of order
            for i, keys in enumerate([[10, 11, 12, 13], [0, 1, 2, 3]]):
                tbl = DataFrame.from_polars(
                    pl.DataFrame({"key": pl.Series(keys, dtype=pl.Int32())}), stream
                ).table
                await ch.send(
                    context,
                    Message(
                        i,
                        TableChunk.from_pylibcudf_table(
                            tbl, stream, exclusive_view=True, br=context.br()
                        ),
                    ),
                )
            await ch.drain(context)

        with reserve_op_id() as op_id:
            _, scheme = await gather_in_task_group(
                _send(),
                extract_orderscheme(
                    context, comm, schema_ir, ir_context, ch, order_keys, op_id
                ),
            )
        return scheme

    assert asyncio.run(_run()) is None


@pytest.mark.spmd
def test_extract_orderscheme_single_chunk(spmd_engine) -> None:
    """One chunk on a single rank → num_partitions == 1 < 2 → None."""
    context = spmd_engine.context
    comm = spmd_engine.comm

    if comm.nranks != 1:
        pytest.skip("single-partition None path only applies when nranks == 1")

    order_keys = [OrderKey(0, plc.types.Order.ASCENDING, plc.types.NullOrder.BEFORE)]
    schema_ir = Empty({"key": DataType(pl.Int32())})
    ir_context = IRExecutionContext(get_cuda_stream=context.get_stream_from_pool)

    async def _run():
        ch = context.create_channel()
        with reserve_op_id() as op_id:
            _, scheme = await gather_in_task_group(
                _send_sorted_chunks(context, ch, key_start=0, n_chunks=1, n_rows=4),
                extract_orderscheme(
                    context, comm, schema_ir, ir_context, ch, order_keys, op_id
                ),
            )
        return scheme

    assert asyncio.run(_run()) is None
