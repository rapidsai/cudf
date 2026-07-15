# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Hybrid scan prefetch pipeline."""

from __future__ import annotations

import asyncio
import ctypes
import threading
from concurrent.futures import Future
from typing import TYPE_CHECKING

import pylibcudf as plc
from rapidsmpf.memory.buffer import MemoryType
from rapidsmpf.memory.pinned_memory_resource import PinnedMemoryResource
from rapidsmpf.streaming.core.memory_reserve_or_wait import reserve_memory
from rmm.pylibrmm.stream import Stream

from cudf_polars.dsl.ir import _prepare_parquet_predicate
from cudf_polars.dsl.to_ast import to_parquet_filter
from cudf_polars.streaming.io import PrefetchedByteRanges, _fetch_byte_ranges

if TYPE_CHECKING:
    from kvikio.cufile import CuFile, IOFuture
    from kvikio.remote_file import RemoteFile

    from rapidsmpf.streaming.core.context import Context

    from cudf_polars.streaming.io import SplitScan


class PinnedBuffer:
    """Pinned host buffer backed by a rapidsmpf PinnedMemoryResource pool."""

    __slots__ = ("mr", "ptr", "nbytes", "stream", "reservation", "array")

    def __init__(
        self,
        mr: PinnedMemoryResource,
        nbytes: int,
        stream: Stream,
        context: Context,
        loop: asyncio.AbstractEventLoop,
    ) -> None:
        self.mr = mr
        self.nbytes = nbytes
        self.stream = stream  # keep alive so __del__ can pass it back to the pool
        self.reservation = asyncio.run_coroutine_threadsafe(
            reserve_memory(context, size=nbytes, net_memory_delta=nbytes, mem_type=MemoryType.PINNED_HOST),
            loop,
        ).result()
        self.ptr = mr.allocate(nbytes, stream)
        self.array = memoryview((ctypes.c_uint8 * nbytes).from_address(self.ptr))

    def __del__(self) -> None:
        self.reservation.clear()
        self.mr.deallocate(self.ptr, self.nbytes, self.stream)


def pread_ranges(
    handle: CuFile | RemoteFile,
    ranges: list[plc.io.text.ByteRangeInfo],
    pinned_mr: PinnedMemoryResource,
    stream: Stream,
    context: Context,
    loop: asyncio.AbstractEventLoop,
) -> tuple[memoryview | None, list[IOFuture], PinnedBuffer | None]:
    """Issue concurrent async reads for each range into a single pinned host buffer."""
    total = sum(r.size for r in ranges)
    if not total:
        return None, [], None
    buf = PinnedBuffer(pinned_mr, total, stream, context, loop)
    futures = []
    offset = 0
    for r in ranges:
        futures.append(
            handle.pread(buf.array[offset : offset + r.size], size=r.size, file_offset=r.offset)
        )
        offset += r.size
    return buf.array, futures, buf


def prefetch_split(
    scan: SplitScan,
    stream: Stream,
    pinned_mr: PinnedMemoryResource,
    context: Context,
    loop: asyncio.AbstractEventLoop,
) -> PrefetchedByteRanges | None:
    """
    Run stats and bloom pruning for one SplitScan and issue async reads.

    Returns None when the split cannot use the hybrid-scan path, signalling
    the producer to fall back to SplitScan.do_evaluate.
    """
    cached_info = scan.cached_parquet_info
    if cached_info is None:
        return None

    row_group_num_rows = cached_info[0].file_metadata.row_group_num_rows
    total_row_groups = len(row_group_num_rows)

    if scan.total_splits > total_row_groups:
        return None

    rg_stride = total_row_groups // scan.total_splits
    skip_rgs = rg_stride * scan.split_index
    end_rg = (
        total_row_groups
        if scan.split_index == scan.total_splits - 1
        else skip_rgs + rg_stride
    )
    row_group_indices = list(range(skip_rgs, end_rg))

    predicate = scan.base_scan.predicate
    if predicate is None:
        return None

    plc_filter = to_parquet_filter(
        _prepare_parquet_predicate(
            predicate.value, scan.paths, scan.schema, scan.base_scan.with_columns
        ),
        stream=stream,
    )
    if plc_filter is None:
        return None

    options = (
        plc.io.parquet.ParquetReaderOptions.builder(plc.io.SourceInfo(scan.paths))
        .decimal_width(plc.TypeId.DECIMAL128)
        .build()
    )
    if scan.base_scan.with_columns is not None:
        options.set_column_names(scan.base_scan.with_columns)
    options.set_filter(plc_filter)

    reader = cached_info[0].hybrid_scan_reader(id(scan.base_scan), options)

    if scan.parquet_options.hybrid_scan_stats_pruning:
        row_group_indices = reader.filter_row_groups_with_stats(
            row_group_indices, options, stream=stream
        )

    if row_group_indices:
        bloom_ranges, _ = reader.secondary_filters_byte_ranges(row_group_indices, options)
        if bloom_ranges:
            bloom_chunks = _fetch_byte_ranges(scan.paths, bloom_ranges, stream)
            row_group_indices = reader.filter_row_groups_with_bloom_filters(
                bloom_chunks, row_group_indices, options, stream=stream
            )

    if not row_group_indices:
        return PrefetchedByteRanges.empty()

    filter_ranges = reader.filter_column_chunks_byte_ranges(row_group_indices, options)
    payload_ranges = reader.payload_column_chunks_byte_ranges(row_group_indices, options)

    handle = cached_info[0].remote_handle()
    filter_host, filter_futures, filter_buf = pread_ranges(handle, filter_ranges, pinned_mr, stream, context, loop)
    payload_host, payload_futures, payload_buf = pread_ranges(handle, payload_ranges, pinned_mr, stream, context, loop)

    return PrefetchedByteRanges(
        row_group_indices=row_group_indices,
        filter_ranges=filter_ranges,
        payload_ranges=payload_ranges,
        filter_host=filter_host,
        payload_host=payload_host,
        filter_futures=filter_futures,
        payload_futures=payload_futures,
        filter_buf=filter_buf,
        payload_buf=payload_buf,
    )


# TODO: Replace with a cucascade::io::datasource that accepts fadvise() hints
# issued before evaluation, so pre-reading is driven by the datasource layer
# rather than a separate host-pinned pipeline.
class HybridScanPrefetcher:
    """
    Prefetch pipeline for SplitScan tasks.

    One worker thread per producer runs stats and bloom pruning and issues
    async reads to pinned host memory ahead of each producer's evaluation
    loop. Pinned memory backpressure is managed via rapidsmpf's reservation
    system.
    """

    def __init__(
        self,
        scans: list[SplitScan],
        num_workers: int,
        context: Context,
        pinned_mr: PinnedMemoryResource | None = None,
    ):
        if pinned_mr is None:
            raise ValueError(
                "HybridScanPrefetcher requires a PinnedMemoryResource; "
                "pass context.br().pinned_mr or enable pinned memory."
            )
        self.pinned_mr = pinned_mr
        self.context = context
        self.loop = asyncio.get_running_loop()
        self.futures: list[Future[PrefetchedByteRanges | None]] = [Future() for _ in scans]

        worker_tasks: list[list[tuple[int, SplitScan]]] = [[] for _ in range(num_workers)]
        for task_idx, scan in enumerate(scans):
            worker_tasks[task_idx % num_workers].append((task_idx, scan))

        self.threads = [
            threading.Thread(
                target=self.run_worker,
                args=(tasks,),
                daemon=True,
                name=f"hybrid-prefetch-{i}",
            )
            for i, tasks in enumerate(worker_tasks)
        ]
        for t in self.threads:
            t.start()

    def run_worker(
        self,
        tasks: list[tuple[int, SplitScan]],
    ) -> None:
        stream = Stream()
        for task_idx, scan in tasks:
            try:
                self.futures[task_idx].set_result(
                    prefetch_split(scan, stream, self.pinned_mr, self.context, self.loop)
                )
            except Exception as exc:
                self.futures[task_idx].set_exception(exc)

    def get(self, task_idx: int) -> PrefetchedByteRanges | None:
        """Block until task_idx's prefetch result is ready and return it."""
        return self.futures[task_idx].result()
