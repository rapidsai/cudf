# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""CUDA stream utilities."""

from __future__ import annotations

from typing import TYPE_CHECKING

from rmm.pylibrmm.stream import DEFAULT_STREAM

if TYPE_CHECKING:
    from collections.abc import Iterable

    from rmm.pylibrmm.stream import Stream


def get_dask_cuda_stream() -> Stream:
    """Get the default CUDA stream for Dask."""
    return DEFAULT_STREAM


def get_cuda_stream() -> Stream:
    """Get the default CUDA stream for the current thread."""
    return DEFAULT_STREAM


def join_cuda_streams(
    *, downstreams: Iterable[Stream], upstreams: Iterable[Stream]
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
    return


def get_joined_cuda_stream(*, upstreams: Iterable[Stream]) -> Stream:
    """
    Return a CUDA stream that is joined to the given streams.

    Parameters
    ----------
    upstreams
        CUDA streams that will be ordered before the returned stream.

    Returns
    -------
    CUDA stream that is joined to the given streams.
    """
    ret = get_cuda_stream()
    join_cuda_streams(downstreams=(ret,), upstreams=upstreams)
    return ret
