# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Utilities for IR nodes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import pylibcudf as plc

from cudf_polars.dsl.tracing import nvtx_annotate_cudf_polars
from cudf_polars.streaming.io import Scan

if TYPE_CHECKING:
    from cudf_polars.streaming.base import StatsCollector


@dataclass(frozen=True)
class CachedParquetInfo:
    """
    Metadata for a parquet file.

    File metadata is only cached when the setting
    ``ParquetOptions.prefetch_file_metadata`` is ``True``. Metadata is cached
    for the duration of the query.

    Parameters
    ----------
    path
        The path of an individual parquet file. This is one element of a
        ``paths`` tuple in a ``Scan`` node.
    size
        The size of the parquet file, in bytes. This is typically only set
        for remote URLs, since it allows skipping subsequent HTTP HEAD requests
        made by kvikio on operations involving that file.
    file_metadata
        The ``FileMetaData`` object for the parquet file returned from
        ``read_parquet_footers``.
    """

    path: str
    size: int | None
    file_metadata: plc.io.parquet_metadata.FileMetaData


@nvtx_annotate_cudf_polars(message="fetch_parquet_footers_for_paths")
def _prefetch_parquet_footers_for_paths(paths: list[str]) -> list[CachedParquetInfo]:
    """
    Prefetch parquet footers for a list of paths.

    This is typically executed concurrently with prefetch operations for other
    path groups for other parquet scan nodes.

    Parameters
    ----------
    paths
        The paths to prefetch.

    Returns
    -------
    paths
        The original input ``paths``.
    metadata
        The list of ``FileMetaData`` objects for the ``paths``.
    """
    # TODO: https://github.com/rapidsai/cudf/issues/22734, use object metadata from polars
    # For now, we'll just use kvikio to explicitly get the size.
    sizes: list[int | None] = []

    try:  # pragma: no cover; kvikio is optional
        import kvikio
    except ImportError:
        kvikio = None

    for path in paths:
        if (
            paths and kvikio is not None and plc.io.SourceInfo._is_remote_uri(path)
        ):  # pragma: no cover; kvikio is optional
            # We're OK to use `kvikio.RemoteFile.open` here. It does make an HTTP HEAD
            # request for S3/HTTP endpoints, but that's the entire reason we're running
            # this code. So long as it makes just *one* HTTP request, there's no advantage
            # to inferring the endpoint type.
            with kvikio.RemoteFile.open(path) as remote_file:
                sizes.append(remote_file.nbytes())
        else:
            sizes.append(None)

    metadata = plc.io.parquet_metadata.read_parquet_footers(
        plc.io.types.SourceInfo(
            [
                plc.io.types.FilepathSource(path, size)
                for path, size in zip(paths, sizes, strict=True)
            ]
        )
    )

    return [
        CachedParquetInfo(path, size, file_metadata)
        for path, size, file_metadata in zip(paths, sizes, metadata, strict=True)
    ]


def _cached_parquet_info_from_stats(
    stats: StatsCollector | None,
) -> dict[str, CachedParquetInfo]:
    """Return path -> cached parquet info seeded from statistics collection."""
    from cudf_polars.streaming.io import ParquetSourceInfo

    cached_parquet_info: dict[str, CachedParquetInfo] = {}
    if stats is None:
        return cached_parquet_info

    for node, datasource_info in stats.scan_stats.items():
        if (
            isinstance(node, Scan)
            and node.typ == "parquet"
            and isinstance(datasource_info, ParquetSourceInfo)
            and datasource_info.cached_parquet_info is not None
        ):
            for info in datasource_info.cached_parquet_info:
                cached_parquet_info[info.path] = info
    return cached_parquet_info


@nvtx_annotate_cudf_polars(message="prefetch_cached_parquet_info_for_paths")
def prefetch_cached_parquet_info_for_paths(
    paths: list[str],
    *,
    stats: StatsCollector | None = None,
) -> list[CachedParquetInfo]:
    """
    Prefetch parquet metadata for a path group.

    Reuses footers already collected during statistics gathering when
    available and fetches any remaining paths.

    Parameters
    ----------
    paths
        Ordered list of parquet file paths for one scan task group.
    stats
        Optional statistics collector with already-cached footers.

    Returns
    -------
    Cached parquet metadata ordered to match ``paths``.
    """
    cached_by_path = _cached_parquet_info_from_stats(stats)
    missing_paths = [path for path in paths if path not in cached_by_path]

    if missing_paths:
        fetched = _prefetch_parquet_footers_for_paths(missing_paths)
        for info in fetched:
            cached_by_path[info.path] = info

    return [cached_by_path[path] for path in paths]
