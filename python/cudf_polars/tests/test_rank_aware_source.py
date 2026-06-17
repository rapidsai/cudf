# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import functools
import sys

import cloudpickle
import pytest

import polars as pl
from polars.io.plugins import register_io_source
from polars.testing import assert_frame_equal

import cudf_polars.streaming.io  # noqa: F401  (registers the PythonScan lower handler)
from cudf_polars.containers import DataFrame, DataType
from cudf_polars.dsl import ir
from cudf_polars.streaming.dispatch import lower_ir_node
from cudf_polars.streaming.rank_aware_source import RankAwareSource, SizedChunks
from cudf_polars.utils.cuda_stream import get_cuda_stream

# The scan-source classes below are defined in this test module. When shipped to
# Ray/Dask workers they would otherwise be pickled by reference, and the worker
# cannot import this test module. Register the module for pickle-by-value so the
# class definitions travel with the instances. Ray vendors its own cloudpickle,
# so register on that too when Ray is installed.
cloudpickle.register_pickle_by_value(sys.modules[__name__])
try:
    from ray import cloudpickle as ray_cloudpickle
except ImportError:
    pass
else:
    ray_cloudpickle.register_pickle_by_value(sys.modules[__name__])


class _Source(RankAwareSource):
    """
    Rank-aware scan source that emits ``df`` on rank 0 only.
    """

    def __init__(self, df: pl.DataFrame) -> None:
        self.df = df

    def __call__(self, with_columns, predicate, n_rows, batch_size, rank=0, nranks=1):
        out = self.df.clear() if rank else self.df
        if with_columns is not None:
            out = out.select(with_columns)
        if predicate is not None:
            out = out.filter(predicate)
        if n_rows is not None:
            out = out.head(n_rows)
        yield out


def test_rank_aware_source(engine: pl.GPUEngine):
    df = pl.DataFrame(
        {"a": pl.Series([1, 2, 3, 4, 5], dtype=pl.Int8()), "b": [10, 20, 30, 40, 50]}
    )
    q = register_io_source(_Source(df), schema={"a": pl.Int8, "b": pl.Int64})
    assert_frame_equal(q.collect(engine=engine), df)
    assert_frame_equal(q.select("a").collect(engine=engine), df.select("a"))
    assert_frame_equal(
        q.filter(pl.col("a") > 2).collect(engine=engine), df.filter(pl.col("a") > 2)
    )


class _MultiChunkSource(RankAwareSource):
    """Rank-aware scan source that emits two chunks on rank 0 only."""

    def __init__(self, chunks: list[pl.DataFrame]) -> None:
        self.chunks = chunks

    def __call__(self, with_columns, predicate, n_rows, batch_size, rank=0, nranks=1):
        # Emit on rank 0 only (the lone worker defaults to rank 0).
        if rank:
            yield from (c.clear() for c in self.chunks)
        else:
            yield from self.chunks


def test_rank_aware_source_multi_chunk(engine: pl.GPUEngine):
    chunks = [pl.DataFrame({"a": [1, 2]}), pl.DataFrame({"a": [3, 4]})]
    q = register_io_source(_MultiChunkSource(chunks), schema={"a": pl.Int64})
    assert_frame_equal(q.collect(engine=engine), pl.DataFrame({"a": [1, 2, 3, 4]}))


class _GpuSource(RankAwareSource):
    """Rank-aware source that yields GPU-resident cudf-polars DataFrames."""

    def __init__(self, df: pl.DataFrame) -> None:
        self.df = df

    def __call__(self, with_columns, predicate, n_rows, batch_size, rank=0, nranks=1):
        df = self.df if not rank else self.df.clear()
        yield DataFrame.from_polars(df, stream=get_cuda_stream())


def test_rank_aware_source_gpu_chunks(engine: pl.GPUEngine):
    df = pl.DataFrame({"a": [1, 2, 3]})
    q = register_io_source(_GpuSource(df), schema={"a": pl.Int64})
    assert_frame_equal(q.collect(engine=engine), df)


class _PartitioningSource(RankAwareSource):
    """Self-partitions a shared frame across workers using ``rank``/``nranks``."""

    def __init__(self, df: pl.DataFrame) -> None:
        self.df = df

    def __call__(self, with_columns, predicate, n_rows, batch_size, rank=0, nranks=1):
        out = self.df.gather_every(nranks, offset=rank)
        if with_columns is not None:
            out = out.select(with_columns)
        yield out


def test_rank_aware_source_self_partition(engine: pl.GPUEngine):
    df = pl.DataFrame({"a": list(range(10))})
    q = register_io_source(_PartitioningSource(df), schema={"a": pl.Int64})
    # Order across ranks is not guaranteed, so compare after sorting.
    assert_frame_equal(q.collect(engine=engine).sort("a"), df)


class _PartialArgSource(RankAwareSource):
    """Source whose extra keyword is supplied via ``functools.partial``."""

    def __call__(
        self, with_columns, predicate, n_rows, batch_size, rank=0, nranks=1, *, value=0
    ):
        out = pl.DataFrame({"a": [value]})
        if rank:
            out = out.clear()
        if with_columns is not None:
            out = out.select(with_columns)
        yield out


def test_rank_aware_source_preserves_partial(engine: pl.GPUEngine):
    # A user may wrap their source in functools.partial. Binding the rank must
    # preserve the partial-applied arguments: if it dropped ``value`` the source
    # would fall back to the default (0) instead of 42.
    source = functools.partial(_PartialArgSource(), value=42)
    q = register_io_source(source, schema={"a": pl.Int64})
    assert_frame_equal(q.collect(engine=engine), pl.DataFrame({"a": [42]}))


def _plain_source(with_columns, predicate, n_rows, batch_size):
    yield pl.DataFrame({"a": [1]})


def _scan_fn(source) -> object:
    """The register_io_source wrapper stored in the PythonScan options."""
    lf = register_io_source(source, schema={"a": pl.Int64})
    return lf._ldf.visit().view_current_node().options[0]


class _LowerState:
    """Minimal stand-in for the lowering transformer (only ``state`` is read)."""

    def __init__(self, nranks: int) -> None:
        self.state = {"nranks": nranks}


def _lower_python_scan(source, nranks: int) -> int:
    schema = {"a": DataType(pl.Int64())}
    # options = (scan_fn, with_columns, source_type)
    node = ir.PythonScan(schema, (_scan_fn(source), None, "io_plugin"), None)
    # _LowerState is a minimal stub, not a full GenericTransformer.
    _, partition_info = lower_ir_node(node, _LowerState(nranks=nranks))  # type: ignore[arg-type]
    return partition_info[node].count


def test_python_scan_partition_count():
    # A PythonScan can emit multiple chunks and the count is unknown at lowering,
    # so it always lowers as multi-partition (count > 1) regardless of world size.
    assert _lower_python_scan(_plain_source, nranks=3) == 2
    assert _lower_python_scan(_plain_source, nranks=1) == 2


def test_sized_chunks_reports_count_and_iterates():
    # SizedChunks reports its length up front (so the streaming engine can stream
    # lazily) and is a single-pass iterator over the wrapped chunks.
    frames = [pl.DataFrame({"a": [1]}), pl.DataFrame({"a": [2]})]
    sized = SizedChunks(2, iter(frames))
    assert len(sized) == 2
    assert list(sized) == frames  # single-pass iteration yields the chunks
    assert list(sized) == []  # exhausted after one pass


@pytest.mark.spmd
def test_rank_aware_source_collect_twice(spmd_engine: pl.GPUEngine):
    df = pl.DataFrame({"a": list(range(10))})
    q = register_io_source(_PartitioningSource(df), schema={"a": pl.Int64})

    # First collect runs the rank-aware source under streaming (binds the rank).
    q.collect(engine=spmd_engine)

    # Second collect uses the default engine, which does not bind a rank, so the
    # source must run with its rank=0, nranks=1 defaults and return everything.
    assert_frame_equal(q.collect().sort("a"), df)
