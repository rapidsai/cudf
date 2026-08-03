# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Hybrid scan prefetch pipeline."""

from __future__ import annotations

import asyncio
import ctypes
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, Any, Self

import cuda.bindings.runtime as cudart

try:  # pragma: no cover; cucascade is optional
    import cucascade
except ImportError:
    cucascade = None

_cucascade_engine: Any | None = None
_cucascade_engine_lock = threading.Lock()

import pylibcudf as plc
from rapidsmpf.memory.buffer import MemoryType
from rapidsmpf.streaming.core.memory_reserve_or_wait import reserve_memory

from cudf_polars.dsl.ir import _prepare_parquet_predicate
from cudf_polars.dsl.to_ast import to_parquet_filter
from cudf_polars.dsl.tracing import nvtx_annotate_cudf_polars
from cudf_polars.streaming.io import PrefetchedByteRanges, _fetch_byte_ranges

if TYPE_CHECKING:
    from concurrent.futures import Future

    from kvikio.cufile import CuFile, IOFuture
    from kvikio.remote_file import RemoteFile

    from rapidsmpf.memory.memory_reservation import MemoryReservation
    from rapidsmpf.memory.pinned_memory_resource import PinnedMemoryResource
    from rapidsmpf.streaming.core.context import Context
    from rmm.pylibrmm.stream import Stream

    from cudf_polars.streaming.io import SplitScan


class PinnedBuffer:
    """Pinned host buffer backed by a rapidsmpf PinnedMemoryResource pool."""

    __slots__ = ("array", "mr", "nbytes", "ptr", "reservation", "stream")

    def __init__(
        self,
        mr: PinnedMemoryResource,
        nbytes: int,
        stream: Stream,
        reservation: MemoryReservation,
    ) -> None:
        self.mr = mr
        self.nbytes = nbytes
        self.stream = stream
        self.reservation = reservation
        self.ptr = mr.allocate(nbytes, stream)
        self.array = memoryview((ctypes.c_uint8 * nbytes).from_address(self.ptr))

    def __del__(self) -> None:  # noqa: D105
        # Guard against partial init.
        if hasattr(self, "reservation"):
            self.reservation.clear()
        if hasattr(self, "ptr"):
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
    # Blocks this worker thread, not the event loop. The loop stays free to
    # run other coroutines while we wait for the reservation.
    with nvtx_annotate_cudf_polars(message="reserve_pinned_memory", payload=total):
        reservation = asyncio.run_coroutine_threadsafe(
            reserve_memory(
                context,
                size=total,
                net_memory_delta=total,
                mem_type=MemoryType.PINNED_HOST,
            ),
            loop,
        ).result()
    buf = PinnedBuffer(pinned_mr, total, stream, reservation)
    futures = []
    offset = 0
    with nvtx_annotate_cudf_polars(message="read_ranges:submit", payload=total):
        for r in ranges:
            futures.append(
                handle.pread(
                    buf.array[offset : offset + r.size],
                    size=r.size,
                    file_offset=r.offset,
                )
            )
            offset += r.size
    return buf.array, futures, buf


def _plan_hybrid_scan_prefetch(
    scan: SplitScan,
    stream: Stream,
) -> PrefetchedByteRanges | None:
    """
    Prune row groups and compute filter/payload byte ranges for one split.

    Parameters
    ----------
    scan
        The split scan to plan.
    stream
        CUDA stream used for filter expression compilation and stats pruning.

    Returns
    -------
    PrefetchedByteRanges | None
        ``None`` when the predicate cannot be expressed as a parquet filter,
        in which case the producer falls back to ``SplitScan.do_evaluate``.
        :meth:`PrefetchedByteRanges.empty` when all row groups are pruned away.
    """
    cached_info = scan.cached_parquet_info
    assert cached_info is not None

    row_group_num_rows = cached_info[0].file_metadata.row_group_num_rows
    total_row_groups = len(row_group_num_rows)

    rg_stride = total_row_groups // scan.total_splits
    skip_rgs = rg_stride * scan.split_index
    end_rg = (
        total_row_groups
        if scan.split_index == scan.total_splits - 1
        else skip_rgs + rg_stride
    )
    row_group_indices = list(range(skip_rgs, end_rg))

    predicate = scan.base_scan.predicate
    assert predicate is not None

    with nvtx_annotate_cudf_polars(message="to_parquet_filter"):
        plc_filter = to_parquet_filter(
            _prepare_parquet_predicate(
                predicate.value, scan.paths, scan.schema, scan.base_scan.with_columns
            ),
            stream=stream,
        )
    if plc_filter is None:
        return None

    with nvtx_annotate_cudf_polars(message="build_reader_options"):
        options = (
            plc.io.parquet.ParquetReaderOptions.builder(
                plc.io.SourceInfo(
                    [
                        plc.io.types.FilepathSource(
                            cached_info[0].path, cached_info[0].size
                        )
                    ]
                )
            )
            .decimal_width(plc.TypeId.DECIMAL128)
            .build()
        )
        if scan.base_scan.with_columns is not None:
            options.set_column_names(scan.base_scan.with_columns)
        options.set_filter(plc_filter)

    with nvtx_annotate_cudf_polars(message="hybrid_scan_reader"):
        reader = cached_info[0].hybrid_scan_reader(options)

    if scan.parquet_options._hybrid_scan_stats_pruning:
        with nvtx_annotate_cudf_polars(message="filter_row_groups_with_stats"):
            row_group_indices = reader.filter_row_groups_with_stats(
                row_group_indices, options, stream=stream
            )

    if row_group_indices:
        bloom_ranges, _ = reader.secondary_filters_byte_ranges(
            row_group_indices, options
        )
        if bloom_ranges:
            with nvtx_annotate_cudf_polars(
                message="filter_row_groups_with_bloom_filters"
            ):
                bloom_chunks = _fetch_byte_ranges(
                    plc.io.SourceInfo(
                        [
                            plc.io.types.FilepathSource(
                                cached_info[0].path, cached_info[0].size
                            )
                        ]
                    ),
                    bloom_ranges,
                    stream,
                )
                row_group_indices = reader.filter_row_groups_with_bloom_filters(
                    bloom_chunks, row_group_indices, options, stream=stream
                )

    if not row_group_indices:
        return PrefetchedByteRanges.empty()

    with nvtx_annotate_cudf_polars(message="byte_range_computation"):
        filter_ranges = reader.filter_column_chunks_byte_ranges(
            row_group_indices, options
        )
        payload_ranges = reader.payload_column_chunks_byte_ranges(
            row_group_indices, options
        )

    # TODO: coalesce nearby ranges before issuing pread calls.
    # https://github.com/rapidsai/cudf/pull/23317#discussion_r3668809937
    return PrefetchedByteRanges(
        row_group_indices=row_group_indices,
        filter_ranges=filter_ranges,
        payload_ranges=payload_ranges,
        filter_host=None,
        payload_host=None,
    )


def prefetch_scan_byte_ranges(
    scan: SplitScan,
    stream: Stream,
    pinned_mr: PinnedMemoryResource,
    context: Context,
    loop: asyncio.AbstractEventLoop,
) -> PrefetchedByteRanges | None:
    """
    Run stats and bloom pruning for one SplitScan and prefetch byte ranges.

    Parameters
    ----------
    scan
        The split scan task to prefetch.
    stream
        CUDA stream used for filter expression compilation.
    pinned_mr
        Pinned memory resource to allocate host buffers from.
    context
        rapidsmpf context.
    loop
        Event loop for the calling async context.

    Returns
    -------
    PrefetchedByteRanges | None
        None when the split cannot use the hybrid-scan path, signalling
        the producer to fall back to SplitScan.do_evaluate.
    """
    planned = _plan_hybrid_scan_prefetch(scan, stream)
    if planned is None or not planned.row_group_indices:
        return planned

    assert scan.cached_parquet_info is not None
    handle = scan.cached_parquet_info[0].remote_handle()
    filter_bytes = sum(r.size for r in planned.filter_ranges)
    payload_bytes = sum(r.size for r in planned.payload_ranges)
    with nvtx_annotate_cudf_polars(
        message=f"pread_filter_and_payload [{scan.split_index + 1}/{scan.total_splits}]:filter={filter_bytes}B,payload={payload_bytes}B"
    ):
        filter_host, filter_futures, filter_buf = pread_ranges(
            handle, planned.filter_ranges, pinned_mr, stream, context, loop
        )
        payload_host, payload_futures, payload_buf = pread_ranges(
            handle, planned.payload_ranges, pinned_mr, stream, context, loop
        )

    return PrefetchedByteRanges(
        row_group_indices=planned.row_group_indices,
        filter_ranges=planned.filter_ranges,
        payload_ranges=planned.payload_ranges,
        filter_host=filter_host,
        payload_host=payload_host,
        filter_futures=filter_futures,
        payload_futures=payload_futures,
        filter_buf=filter_buf,
        payload_buf=payload_buf,
    )


def fadvise_scan_byte_ranges(
    scan: SplitScan,
    stream: Stream,
    datasource_cache: dict[str, Any],
    dev_id: int,
    pinned_mr: PinnedMemoryResource,
    context: Context,
    loop: asyncio.AbstractEventLoop,
) -> PrefetchedByteRanges | None:
    """
    Run stats and bloom pruning for one SplitScan and prefetch byte ranges.

    Parameters
    ----------
    scan
        The split scan task to prefetch.
    stream
        CUDA stream used for filter expression compilation.
    datasource_cache
        Per-query cache mapping file path to its open datasource.
    dev_id
        CUDA device id for staging.
    pinned_mr
        Pinned memory resource to allocate host buffers from.
    context
        rapidsmpf context.
    loop
        Event loop for the calling async context.

    Returns
    -------
    PrefetchedByteRanges | None
        None when the split cannot use the hybrid-scan path, signalling
        the producer to fall back to SplitScan.do_evaluate.
    """
    planned = _plan_hybrid_scan_prefetch(scan, stream)
    if planned is None or not planned.row_group_indices:
        return planned

    datasource = datasource_cache[scan.paths[0]].duplicate()

    filter_bytes = sum(r.size for r in planned.filter_ranges)
    payload_bytes = sum(r.size for r in planned.payload_ranges)
    all_ranges = [
        (r.offset, r.size) for r in planned.filter_ranges + planned.payload_ranges
    ]

    with nvtx_annotate_cudf_polars(
        message=f"fadvise [{scan.split_index + 1}/{scan.total_splits}]:filter={filter_bytes}B,payload={payload_bytes}B"
    ):
        datasource.fadvise(all_ranges, dev_id)

    # TODO: eliminate the extra copy (cuCascade bounce buffer to rapidsmpf PinnedBuffer
    # to GPU) by having cuCascade expose the staged data as a device buffer directly.
    filter_buf = PinnedBuffer(pinned_mr, filter_bytes, stream, context, loop)
    payload_buf = PinnedBuffer(pinned_mr, payload_bytes, stream, context, loop)

    filter_future = datasource.read_all_ranges_async(
        [(r.offset, r.size) for r in planned.filter_ranges], filter_buf.array
    )
    payload_future = datasource.read_all_ranges_async(
        [(r.offset, r.size) for r in planned.payload_ranges], payload_buf.array
    )

    return PrefetchedByteRanges(
        row_group_indices=planned.row_group_indices,
        filter_ranges=planned.filter_ranges,
        payload_ranges=planned.payload_ranges,
        filter_host=filter_buf.array,
        payload_host=payload_buf.array,
        filter_futures=[filter_future],
        payload_futures=[payload_future],
        filter_buf=filter_buf,
        payload_buf=payload_buf,
    )


def _get_cucascade_engine(
    path: str,
    pool_capacity: int | None,
    n_reactors: int | None,
    max_connections: int | None,
    chunk_size: int | None,
    max_n_chunks: int | None,
    enable_cache: bool = False,
) -> Any:
    global _cucascade_engine
    if _cucascade_engine is not None:
        return _cucascade_engine
    with _cucascade_engine_lock:
        if _cucascade_engine is None:
            kwargs: dict[str, Any] = {}
            if pool_capacity is not None:
                kwargs["pool_capacity"] = pool_capacity
            if n_reactors is not None:
                kwargs["n_reactors"] = n_reactors
            if max_connections is not None:
                kwargs["max_connections"] = max_connections
            if chunk_size is not None:
                kwargs["chunk_size"] = chunk_size
            if max_n_chunks is not None:
                kwargs["max_n_chunks"] = max_n_chunks
            kwargs["enable_cache"] = enable_cache
            if plc.io.SourceInfo._is_remote_uri(path):
                # TODO: replace with cucascade.RestEngine.from_environment() once
                # cuCascade exposes a factory that reads standard AWS env vars directly.
                _cucascade_engine = cucascade.RestEngine(
                    access_key_id=os.environ.get("AWS_ACCESS_KEY_ID", ""),
                    secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY", ""),
                    session_token=os.environ.get("AWS_SESSION_TOKEN", ""),
                    region=os.environ.get("AWS_DEFAULT_REGION", "us-east-1"),
                    endpoint=os.environ.get("AWS_ENDPOINT_URL", ""),
                    **kwargs,
                )
            else:
                _cucascade_engine = cucascade.UringEngine(**kwargs)
    return _cucascade_engine


class HybridScanPrefetchExecutor:
    """
    Prefetch executor for SplitScan tasks.

    Submits prefetch work for all splits upfront. Use as a context manager.
    """

    thread_local: threading.local = threading.local()

    @staticmethod
    def init_stream() -> None:
        """Initialise a per-thread CUDA stream."""
        HybridScanPrefetchExecutor.thread_local.stream = Stream()

    def __init__(
        self,
        futures: list[Future[PrefetchedByteRanges | None]],
        executor: ThreadPoolExecutor,
        engine: Any = None,
        datasource_cache: dict[str, Any] | None = None,
    ) -> None:
        self.futures = futures
        self.executor = executor
        self.engine = engine
        self.datasource_cache = datasource_cache or {}

    @classmethod
    def from_scans(
        cls,
        scans: list[SplitScan],
        num_workers: int,
        context: Context,
        prefetch_backend: str,
        cucascade_pool_capacity: int | None = None,
        cucascade_n_reactors: int | None = None,
        cucascade_max_connections: int | None = None,
        cucascade_chunk_size: int | None = None,
        cucascade_max_n_chunks: int | None = None,
        cucascade_enable_cache: bool = False,
    ) -> Self:
        """
        Submit prefetch tasks for all scans.

        Parameters
        ----------
        scans
            Tasks to prefetch.
        num_workers
            Number of background worker threads.
        context
            rapidsmpf context. Pinned memory must be enabled.
        prefetch_backend
            ``"kvikio"`` or ``"cucascade"``.
        cucascade_pool_capacity
            Size in bytes of the cuCascade pinned host memory pool.
        cucascade_n_reactors
            Number of IO reactor threads in the cuCascade engine.

        Returns
        -------
        HybridScanPrefetchExecutor

        Raises
        ------
        ValueError
            If pinned memory is required but not available.
        ImportError
            If ``prefetch_backend`` is ``"cucascade"`` and the ``cucascade``
            package is not installed.
        """
        # TODO: Consider reusing ir_context.py_executor instead of a dedicated pool.
        executor = ThreadPoolExecutor(
            max_workers=num_workers,
            initializer=cls.init_stream,
            thread_name_prefix="hybrid-prefetch",
        )

        if prefetch_backend == "cucascade":
            if cucascade is None:
                raise ImportError(
                    "prefetch_backend='cucascade' requires the cucascade package"
                )
            first_path = scans[0].paths[0] if scans else ""
            engine = _get_cucascade_engine(
                first_path,
                cucascade_pool_capacity,
                cucascade_n_reactors,
                cucascade_max_connections,
                cucascade_chunk_size,
                cucascade_max_n_chunks,
                cucascade_enable_cache,
            )

            _, dev_id = cudart.cudaGetDevice()

            pinned_mr = context.br().pinned_mr
            if pinned_mr is None:
                raise ValueError(
                    "prefetch_backend='cucascade' requires a PinnedMemoryResource; "
                    "enable pinned memory via --pinned-memory."
                )
            loop = asyncio.get_running_loop()

            datasource_cache: dict[str, Any] = {}
            for scan in scans:
                path = scan.paths[0]
                if path not in datasource_cache:
                    datasource_cache[path] = engine.open(path)

            def task(s: SplitScan) -> PrefetchedByteRanges | None:
                return fadvise_scan_byte_ranges(
                    s, cls.thread_local.stream, datasource_cache, dev_id,
                    pinned_mr, context, loop,
                )

        else:
            datasource_cache = {}
            pinned_mr = context.br().pinned_mr
            if pinned_mr is None:
                raise ValueError(
                    "HybridScanPrefetchExecutor requires a PinnedMemoryResource; "
                    "enable pinned memory via --pinned-memory."
                )
            loop = asyncio.get_running_loop()
            stream_pool = context.br().stream_pool

            def task(s: SplitScan) -> PrefetchedByteRanges | None:
                return prefetch_scan_byte_ranges(
                    s, stream_pool.get_stream(), pinned_mr, context, loop
                )

        futures = [executor.submit(task, scan) for scan in scans]
        return cls(
            futures,
            executor,
            engine=engine if prefetch_backend == "cucascade" else None,
            datasource_cache=datasource_cache
            if prefetch_backend == "cucascade"
            else None,
        )

    def __enter__(self) -> Self:
        """Enter the context manager."""
        return self

    def __exit__(self, *args: Any) -> None:
        """Shut down the thread pool, cancelling pending futures."""
        self.executor.shutdown(cancel_futures=True, wait=True)
        self.futures.clear()
        self.datasource_cache.clear()
        self.engine = None

    def result(self, task_idx: int) -> PrefetchedByteRanges | None:
        """Block until the prefetch result for ``task_idx`` is ready."""
        return self.futures[task_idx].result()
