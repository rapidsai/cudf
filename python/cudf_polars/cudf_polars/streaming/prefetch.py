# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Hybrid scan prefetch pipeline."""

from __future__ import annotations

import ctypes
import threading
from typing import TYPE_CHECKING, Any

import numpy as np
import pylibcudf as plc
from rapidsmpf.memory.pinned_memory_resource import PinnedMemoryResource
from rmm.pylibrmm.stream import Stream

from cudf_polars.dsl.ir import _prepare_parquet_predicate
from cudf_polars.dsl.to_ast import to_parquet_filter
from cudf_polars.streaming.io import PrefetchedSplit, _fetch_byte_ranges

if TYPE_CHECKING:
    from cudf_polars.streaming.io import SplitScan


UNSET: Any = object()


class PinnedBuffer:
    """Pinned host buffer backed by a rapidsmpf PinnedMemoryResource pool."""

    __slots__ = ("mr", "ptr", "nbytes", "stream", "array")

    def __init__(self, mr: PinnedMemoryResource, nbytes: int, stream: Stream) -> None:
        self.mr = mr
        self.nbytes = nbytes
        self.stream = stream  # keep alive so __del__ can pass it back to the pool
        self.ptr = mr.allocate(nbytes, stream)
        self.array = np.frombuffer(
            (ctypes.c_uint8 * nbytes).from_address(self.ptr), dtype=np.uint8
        )

    def __del__(self) -> None:
        self.mr.deallocate(self.ptr, self.nbytes, self.stream)


def pread_ranges(
    handle: Any,
    ranges: list[plc.io.text.ByteRangeInfo],
    pinned_mr: PinnedMemoryResource,
    stream: Stream,
) -> tuple[np.ndarray | None, list[Any], PinnedBuffer | None]:
    """Issue concurrent async reads for each range into a single pinned host buffer."""
    total = sum(r.size for r in ranges)
    if not total:
        return None, [], None
    buf = PinnedBuffer(pinned_mr, total, stream)
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
    sem: threading.Semaphore,
    pinned_mr: PinnedMemoryResource,
) -> PrefetchedSplit | None:
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

    reader = plc.io.experimental.HybridScanReader.from_metadata(
        cached_info[0].hybrid_scan_metadata(options)
    )

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
        return PrefetchedSplit(
            row_group_indices=[],
            filter_ranges=[],
            payload_ranges=[],
            filter_host=None,
            payload_host=None,
            filter_futures=[],
            payload_futures=[],
        )

    filter_ranges = reader.filter_column_chunks_byte_ranges(row_group_indices, options)
    payload_ranges = reader.payload_column_chunks_byte_ranges(row_group_indices, options)

    sem.acquire()
    handle = cached_info[0].remote_handle()
    filter_host, filter_futures, filter_buf = pread_ranges(handle, filter_ranges, pinned_mr, stream)
    payload_host, payload_futures, payload_buf = pread_ranges(handle, payload_ranges, pinned_mr, stream)

    return PrefetchedSplit(
        row_group_indices=row_group_indices,
        filter_ranges=filter_ranges,
        payload_ranges=payload_ranges,
        filter_host=filter_host,
        payload_host=payload_host,
        filter_futures=filter_futures,
        payload_futures=payload_futures,
        _sem=sem,
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
    loop. A per-worker semaphore of size ``depth`` limits in-flight pinned
    allocations.
    """

    def __init__(
        self,
        scans: list[SplitScan],
        num_workers: int,
        depth: int = 2,
        pinned_mr: PinnedMemoryResource | None = None,
    ):
        if pinned_mr is None:
            raise ValueError(
                "HybridScanPrefetcher requires a PinnedMemoryResource; "
                "pass context.br().pinned_mr or enable pinned memory."
            )
        self.pinned_mr = pinned_mr
        self.results: list[Any] = [UNSET] * len(scans)
        self.events: list[threading.Event] = [threading.Event() for _ in scans]

        worker_tasks: list[list[tuple[int, SplitScan]]] = [[] for _ in range(num_workers)]
        for task_idx, scan in enumerate(scans):
            worker_tasks[task_idx % num_workers].append((task_idx, scan))

        self.threads = [
            threading.Thread(
                target=self.run_worker,
                args=(tasks, threading.Semaphore(depth)),
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
        sem: threading.Semaphore,
    ) -> None:
        stream = Stream()
        for task_idx, scan in tasks:
            try:
                result: PrefetchedSplit | None = prefetch_split(
                    scan, stream, sem, self.pinned_mr
                )
            except BaseException as exc:
                result = PrefetchedSplit(
                    row_group_indices=[],
                    filter_ranges=[],
                    payload_ranges=[],
                    filter_host=None,
                    payload_host=None,
                    filter_futures=[],
                    payload_futures=[],
                    error=exc,
                )
            self.results[task_idx] = result
            self.events[task_idx].set()

    def get(self, task_idx: int) -> PrefetchedSplit | None:
        """Block until task_idx's prefetch result is ready and return it."""
        self.events[task_idx].wait()
        result = self.results[task_idx]
        if isinstance(result, PrefetchedSplit) and result.error is not None:
            raise result.error
        return result  # type: ignore[return-value]

    def shutdown(self) -> None:
        """Join all worker threads (non-blocking; workers are daemon threads)."""
        for t in self.threads:
            t.join(timeout=0)
