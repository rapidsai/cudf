# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""CUDA stream utilities."""

from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING

import pylibcudf as plc
from rmm.pylibrmm.stream import DEFAULT_STREAM, Stream

if TYPE_CHECKING:
    from collections.abc import Callable, Generator, Sequence

    from cudf_polars.containers import DataFrame
    from cudf_polars.dsl.ir import IRExecutionContext


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


@contextlib.contextmanager
def deferred_dealloc_stream(
    context: IRExecutionContext, dfs: Sequence[DataFrame]
) -> Generator[Stream, None, None]:
    """
    Get a joined CUDA stream, with stream ordering for safe deallocation of inputs.

    Parameters
    ----------
    context
        The execution context, which is used to get the new CUDA stream.
    dfs
        The dataframes to join.

    Yields
    ------
    CUDA stream that is joined to the given streams.

    Notes
    -----
    The interaction between Python's refcounting and CUDA streams is tricky. In general,
    deallocation (when an object's refcount reaches zero) of our objects is
    a stream ordered operation. ``DataFrame.__del__`` will eventually deallocate some
    RMM memory on some stream.

    .. code-block::

       x = ...      # valid on some stream A
       y = func(x)  # valid on some stream B
       del x        # deallocates x *on stream A*

    We need to ensure that the deallocation happens *after* the operation on stream B completes,
    i.e. it needs to be downstream of ``func(x)`` completing.

    To accomplish this, we provide this context manager. You provide the inputs (typically the
    stream-ordered objects that are passed into some function), and we ensure that:

    1. The inputs are all valid on the stream yielded by entering
       the context manager.
    2. The deallocation of the inputs happens after the result is ready
       on the stream yielded by entering the context manager.
    """
    # ensure that the inputs are downstream of result_stream (i.e. valid on result_stream)
    result_stream = get_joined_cuda_stream(
        context.get_cuda_stream, upstreams=[df.stream for df in dfs]
    )

    yield result_stream

    # ensure that the inputs are downstream of result_stream (so that deallocation happens after the result is ready)
    join_cuda_streams(downstreams=[df.stream for df in dfs], upstreams=[result_stream])
