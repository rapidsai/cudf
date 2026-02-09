# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""CUDA stream utilities."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pylibcudf as plc
from rmm.pylibrmm.stream import DEFAULT_STREAM, Stream

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence


def get_dask_cuda_stream() -> Stream:
    """Get the default CUDA stream for Dask."""
    return DEFAULT_STREAM


def get_cuda_stream() -> Stream:
    """Get the default CUDA stream for the current thread."""
    return DEFAULT_STREAM


def get_new_cuda_stream() -> Stream:
    """Get a new CUDA stream for the current thread."""
    return Stream()


def join_cuda_streams(
    *, downstreams: Sequence[Stream], upstreams: Sequence[Stream]
) -> None:
    """
    Join multiple CUDA streams.

    Parameters
    ----------
    downstreams
        CUDA streams to that will be ordered after ``upstreams``.
    upstreams
        CUDA streams that will be ordered before ``downstreams``.
    """
    upstreams = list(upstreams)
    downstreams = list(downstreams)
    for downstream in downstreams:
        plc.experimental.join_streams(upstreams, downstream)


def get_joined_cuda_stream(
    get_cuda_stream: Callable[[], Stream], *, upstreams: Sequence[Stream]
) -> Stream:
    """
    Return a CUDA stream that is joined to the given streams.

    Parameters
    ----------
    get_cuda_stream
        A zero-argument callable that returns a CUDA stream.
    upstreams
        CUDA streams that will be ordered before the returned stream.

    Returns
    -------
    CUDA stream that is joined to the given streams.
    """
    downstream = get_cuda_stream()
    join_cuda_streams(downstreams=(downstream,), upstreams=upstreams)
    return downstream
