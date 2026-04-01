# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""
Asynchronous GPU timing for IR tracing using CUDA events.

To expose *device* side timing in trace events, we use CUDA Events to record the
start and end times of work done on some CUDA stream. See
https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/asynchronous-execution.html#timing-operations-in-cuda-streams
for details.

To extract the timing information and get it back to Python for logging,
we use `cudaLaunchHostFunc` to schedule a callback to run once the work on the
stream is done. There are some restrictions on this callback:

1. It must not call the CUDA API (a requirement of the CUDA runtime).
2. It should do as little work as possible (to avoid deadlocks).

The callback passed to `cudaLaunchHostFunc` is a C function pointer, but
we need to get the actual logs all the way back to Python. We use ctypes
to register the callback function. It simply forwards the token identifying
the task on to our actual worker thread, which extracts the timing information
from the CUDA runtime and logs it.
"""

from __future__ import annotations

import ctypes
import functools
import itertools
import queue
import threading
from typing import Any, TypedDict

from cuda.bindings import runtime

import rmm.pylibrmm.stream

from cudf_polars.dsl.tracing import Scope as _TracingScope
from cudf_polars.utils.cuda_stream import join_cuda_streams


class _GpuTracePending(TypedDict):
    """Metadata and CUDA events for one completed GPU trace interval."""

    ev_start: runtime.cudaEvent_t
    ev_end: runtime.cudaEvent_t
    trace_event_id: str
    query_id: str
    ir_type: str


_SCOPE_GPU = _TracingScope.EVALUATE_IR_NODE_GPU.value

# Lock protecting
# 1. Worker thread creation
# 2. All interactions with _PENDING
_LOCK = threading.Lock()
# Map of token to event pair and metadata
_PENDING: dict[int, _GpuTracePending] = {}
# Counter for unique tokens
_IDS = itertools.count(1)
_QUEUE: queue.Queue[int | None] = queue.Queue()
_WORKER_THREAD: threading.Thread | None = None

# Keep references so ctypes does not collect the callback.
_HOST_CB_TYPE = ctypes.CFUNCTYPE(None, ctypes.c_void_p)


def _raw_host_fn(userdata: ctypes.c_void_p) -> None:
    """Enqueue-only: no CUDA API calls."""
    _QUEUE.put(int(userdata))


_REGISTERED_HOST_FN = _HOST_CB_TYPE(_raw_host_fn)
_HOST_FN = runtime.cudaHostFn_t(ctypes.cast(_REGISTERED_HOST_FN, ctypes.c_void_p).value)


@functools.cache
def get_host_notify_stream() -> rmm.pylibrmm.stream.Stream:
    """
    Return a process-wide CUDA stream used only for GPU-trace ``cudaLaunchHostFunc``.

    The stream is created once (non-blocking) and reused so we do not enqueue Python
    host callbacks on ``result.stream`` (which can deadlock with GIL + stream sync).
    """
    return rmm.pylibrmm.stream.Stream(
        flags=rmm.pylibrmm.stream.CudaStreamFlags.NON_BLOCKING
    )


def rmm_stream_to_cuda_stream_t(
    stream: rmm.pylibrmm.stream.Stream,
) -> runtime.cudaStream_t:
    """Return the ``cudaStream_t`` handle used by ``stream`` for CUDA runtime calls."""
    handle = stream.__cuda_stream__()[1]
    return runtime.cudaStream_t(handle)


def _worker_loop(log: Any) -> None:
    """Drain GPU trace completions; may call CUDA runtime (not in cudaLaunchHostFunc)."""
    while True:
        token = _QUEUE.get()
        if token is None:
            break
        with _LOCK:
            pending = _PENDING.pop(token, None)
        if pending is None:  # pragma: no cover
            log.warning(
                "GPU trace completion had no pending entry (queue out of sync)",
                scope=_SCOPE_GPU,
                token=token,
            )
            continue
        ev_start = pending["ev_start"]
        ev_end = pending["ev_end"]
        try:
            err, elapsed_ms = runtime.cudaEventElapsedTime(ev_start, ev_end)
            if err != runtime.cudaError_t.cudaSuccess:
                log.warning(
                    "Execute IR GPU",
                    scope=_SCOPE_GPU,
                    trace_event_id=pending["trace_event_id"],
                    query_id=pending["query_id"],
                    type=pending["ir_type"],
                    gpu_timing_error=str(err),
                )
            else:
                gpu_duration_ns = int(elapsed_ms * 1_000_000)
                log.info(
                    "Execute IR GPU",
                    scope=_SCOPE_GPU,
                    trace_event_id=pending["trace_event_id"],
                    query_id=pending["query_id"],
                    type=pending["ir_type"],
                    gpu_duration_ns=gpu_duration_ns,
                )
        finally:
            runtime.cudaEventDestroy(ev_start)
            runtime.cudaEventDestroy(ev_end)


def _ensure_worker(log: Any) -> None:
    global _WORKER_THREAD  # noqa: PLW0603 -- singleton background worker
    if _WORKER_THREAD is not None:
        return
    with _LOCK:
        if _WORKER_THREAD is not None:
            return
        t = threading.Thread(
            target=_worker_loop,
            args=(log,),
            name="cudf-polars-gpu-trace",
            daemon=True,
        )
        t.start()
        _WORKER_THREAD = t


def begin_gpu_interval(
    stream: rmm.pylibrmm.stream.Stream,
) -> tuple[runtime.cudaEvent_t, runtime.cudaEvent_t] | None:
    """
    Create two events and record the start event on ``stream``.

    The end event is recorded on ``interval_end_stream`` passed to
    :func:`enqueue_gpu_trace_completion` (typically ``result.stream``).
    On failure before enqueue, call :func:`destroy_event_pair`.
    """
    c_stream = rmm_stream_to_cuda_stream_t(stream)

    err, ev_start = runtime.cudaEventCreate()
    if err != runtime.cudaError_t.cudaSuccess:
        return None
    err, ev_end = runtime.cudaEventCreate()
    if err != runtime.cudaError_t.cudaSuccess:
        runtime.cudaEventDestroy(ev_start)
        return None
    (rerr,) = runtime.cudaEventRecord(ev_start, c_stream)
    if rerr != runtime.cudaError_t.cudaSuccess:
        runtime.cudaEventDestroy(ev_start)
        runtime.cudaEventDestroy(ev_end)
        return None
    return ev_start, ev_end


def enqueue_gpu_trace_completion(
    *,
    interval_end_stream: rmm.pylibrmm.stream.Stream,
    host_notify_stream: rmm.pylibrmm.stream.Stream,
    ev_start: runtime.cudaEvent_t,
    ev_end: runtime.cudaEvent_t,
    trace_event_id: str,
    query_id: str,
    ir_type: str,
    log: Any,
) -> tuple[bool, str | None]:
    """
    Finish the GPU interval and schedule the trace host callback.

    Records ``ev_end`` on ``interval_end_stream`` (which must be downstream of
    the work in IR.do_evaluate), then joins ``host_notify_stream`` downstream of
    ``interval_end_stream`` so ``cudaLaunchHostFunc`` runs only after that work
    (and the end event record) without enqueueing the callback on the result
    stream.

    Returns (success, error_message).
    """
    _ensure_worker(log)
    end_c = rmm_stream_to_cuda_stream_t(interval_end_stream)
    (rerr,) = runtime.cudaEventRecord(ev_end, end_c)
    if rerr != runtime.cudaError_t.cudaSuccess:
        runtime.cudaEventDestroy(ev_start)
        runtime.cudaEventDestroy(ev_end)
        return False, f"cudaEventRecord(end): {rerr}"

    join_cuda_streams(
        downstreams=(host_notify_stream,),
        upstreams=(interval_end_stream,),
    )

    with _LOCK:
        token = next(_IDS)
        _PENDING[token] = {
            "ev_start": ev_start,
            "ev_end": ev_end,
            "trace_event_id": trace_event_id,
            "query_id": query_id,
            "ir_type": ir_type,
        }

    notify_c = rmm_stream_to_cuda_stream_t(host_notify_stream)
    (herr,) = runtime.cudaLaunchHostFunc(notify_c, _HOST_FN, token)
    if herr != runtime.cudaError_t.cudaSuccess:
        with _LOCK:
            _PENDING.pop(token, None)
        runtime.cudaEventDestroy(ev_start)
        runtime.cudaEventDestroy(ev_end)
        return False, f"cudaLaunchHostFunc: {herr}"
    return True, None


def destroy_event_pair(
    ev_start: runtime.cudaEvent_t, ev_end: runtime.cudaEvent_t
) -> None:
    """Destroy a pair of CUDA events after they are no longer needed."""
    runtime.cudaEventDestroy(ev_start)
    runtime.cudaEventDestroy(ev_end)
