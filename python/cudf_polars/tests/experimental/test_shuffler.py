# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio

import pytest
from rapidsmpf.streaming.cudf.channel_metadata import (
    ChannelMetadata,
    HashScheme,
    Partitioning,
)
from rapidsmpf.streaming.cudf.table_chunk import TableChunk

import polars as pl

from cudf_polars.containers import DataFrame, DataType
from cudf_polars.experimental.rapidsmpf.collectives.common import reserve_op_id
from cudf_polars.experimental.rapidsmpf.collectives.shuffle import (
    LocalRepartitioner,
    ShuffleManager,
)
from cudf_polars.experimental.rapidsmpf.frontend.options import StreamingOptions
from cudf_polars.experimental.rapidsmpf.frontend.spmd import allgather_polars_dataframe
from cudf_polars.experimental.rapidsmpf.utils import (
    _is_already_partitioned,
)
from cudf_polars.testing.asserts import assert_gpu_result_equal


@pytest.mark.parametrize(
    "options",
    [
        StreamingOptions(max_rows_per_partition=1, broadcast_join_limit=2),
        StreamingOptions(max_rows_per_partition=5, broadcast_join_limit=2),
    ],
)
def test_join_rapidsmpf(streaming_engine_factory, options) -> None:
    streaming_engine = streaming_engine_factory(options)
    left = pl.LazyFrame(
        {
            "x": range(15),
            "y": [1, 2, 3] * 5,
            "z": [1.0, 2.0, 3.0, 4.0, 5.0] * 3,
        }
    )
    right = pl.LazyFrame(
        {
            "xx": range(6),
            "y": [2, 4, 3] * 2,
            "zz": [1, 2] * 3,
        }
    )
    q = left.join(right, on="y", how="inner")
    assert_gpu_result_equal(q, engine=streaming_engine, check_row_order=False)


@pytest.mark.parametrize(
    "options",
    [
        StreamingOptions(max_rows_per_partition=1),
        StreamingOptions(max_rows_per_partition=5),
    ],
)
def test_sort_rapidsmpf(streaming_engine_factory, options) -> None:
    streaming_engine = streaming_engine_factory(options)
    df = pl.LazyFrame(
        {
            "x": range(15),
            "y": [1, 2, 3] * 5,
            "z": [1.0, 2.0, 3.0, 4.0, 5.0] * 3,
        }
    )
    q = df.sort(by=["y", "z"])
    assert_gpu_result_equal(q, engine=streaming_engine, check_row_order=True)


def test_is_already_partitioned():
    # Unit test for _is_already_partitioned helper
    chunks = 4
    columns = (0, 1)
    modulus = 8
    nranks = 1

    # Exact match: should return True
    metadata_match = ChannelMetadata(
        chunks,
        partitioning=Partitioning(
            inter_rank=HashScheme(columns, modulus),
            local="inherit",
        ),
    )
    assert _is_already_partitioned(metadata_match, columns, modulus, nranks) is True

    # Different columns: should return False
    metadata_diff_cols = ChannelMetadata(
        chunks,
        partitioning=Partitioning(
            inter_rank=HashScheme((0,), modulus),
            local="inherit",
        ),
    )
    assert (
        _is_already_partitioned(metadata_diff_cols, columns, modulus, nranks) is False
    )

    # Different local partitioning: should return False
    metadata_diff_local = ChannelMetadata(
        chunks,
        partitioning=Partitioning(
            inter_rank=HashScheme(columns, modulus),
            local=None,
        ),
    )
    assert (
        _is_already_partitioned(metadata_diff_local, columns, modulus, nranks) is False
    )

    # Different modulus: should return False
    metadata_diff_mod = ChannelMetadata(
        chunks,
        partitioning=Partitioning(
            inter_rank=HashScheme(columns, 16),
            local="inherit",
        ),
    )
    assert _is_already_partitioned(metadata_diff_mod, columns, modulus, nranks) is False

    # No partitioning: should return False
    metadata_none = ChannelMetadata(chunks)
    assert _is_already_partitioned(metadata_none, columns, modulus, nranks) is False

    # Local not "inherit": should return False
    metadata_local = ChannelMetadata(
        chunks,
        partitioning=Partitioning(
            inter_rank=HashScheme(columns, modulus),
            local=HashScheme((0,), 4),
        ),
    )
    assert _is_already_partitioned(metadata_local, columns, modulus, nranks) is False


@pytest.mark.spmd
@pytest.mark.parametrize("local_count", [1, 2, 4])
def test_local_repartitioner_hash(spmd_engine, local_count) -> None:
    # Test LocalRepartitioner with hash partitioning
    context = spmd_engine.context
    comm = spmd_engine.comm

    # 12 rows, 4 distinct key values (3 rows per key)
    pl_df = pl.DataFrame({"key": list(range(4)) * 3, "val": list(range(12))})
    col_names = pl_df.columns
    dtypes = [DataType(dt) for dt in pl_df.dtypes]

    results: list[tuple[int, pl.DataFrame]] = []

    async def _run() -> None:
        stream = context.get_stream_from_pool()
        cudf_df = DataFrame.from_polars(pl_df, stream)
        with reserve_op_id() as op_id:
            shuffle = ShuffleManager(
                context, comm, num_partitions=comm.nranks, collective_id=op_id
            )
            async with shuffle.inserting() as inserter:
                inserter.insert_hash(
                    TableChunk.from_pylibcudf_table(
                        cudf_df.table, stream, exclusive_view=True, br=context.br()
                    ),
                    columns_to_hash=(0,),
                )

            local = LocalRepartitioner(shuffle, local_count=local_count)
            await local.insert_chunks_hash(columns_to_hash=(0,), stream=stream)

            for pid in local.local_partitions():
                tbl = local.extract_chunk(pid, stream)
                results.append(
                    (
                        pid,
                        DataFrame.from_table(
                            tbl, col_names, dtypes, stream
                        ).to_polars(),
                    )
                )

    asyncio.run(_run())

    # Each rank must produce exactly local_count output partitions.
    assert len(results) == local_count

    # Hash is deterministic locally: same key always lands in the same local partition.
    key_to_pid: dict[int, int] = {}
    for pid, df in results:
        for key_val in df["key"].to_list():
            assert key_to_pid.setdefault(key_val, pid) == pid

    # AllGather across ranks to verify global invariants.
    local_df = (
        pl.concat([df for _, df in results])
        if results
        else pl.DataFrame(schema={"key": pl.Int32, "val": pl.Int32})
    )
    with reserve_op_id() as op_id:
        global_df = allgather_polars_dataframe(
            engine=spmd_engine, local_df=local_df, op_id=op_id
        )

    # Every rank inserts 12 rows; all must survive the shuffle + local repartition.
    assert global_df.height == 12 * comm.nranks

    # All 4 distinct key values must appear somewhere across all ranks.
    assert set(global_df["key"].unique().to_list()) == {0, 1, 2, 3}
