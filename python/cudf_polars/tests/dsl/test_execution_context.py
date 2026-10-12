# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import asyncio
import contextvars
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest

from cudf_polars.dsl.ir import IRExecutionContext


@pytest.mark.parametrize("cancellations", [1, 3])
@pytest.mark.parametrize("worker_raises", [False, True])
def test_to_thread_cancellation_drains_worker(cancellations, worker_raises):
    async def run(executor):
        context = IRExecutionContext(executor)
        loop = asyncio.get_running_loop()
        started = asyncio.Event()
        release = threading.Event()
        finished = threading.Event()
        scope_alive = True

        def worker():
            loop.call_soon_threadsafe(started.set)
            assert release.wait(timeout=10)
            assert scope_alive
            finished.set()
            if worker_raises:
                raise ValueError("worker failed during cancellation")

        async def caller():
            nonlocal scope_alive
            try:
                await context.to_thread(worker)
            finally:
                scope_alive = False

        task = asyncio.create_task(caller())
        await asyncio.wait_for(started.wait(), timeout=10)
        try:
            for _ in range(cancellations):
                task.cancel()
                await asyncio.sleep(0)
                assert not task.done()
                assert scope_alive
        finally:
            release.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert finished.is_set()
        assert not scope_alive

    with ThreadPoolExecutor(max_workers=1) as executor:
        asyncio.run(run(executor))


def test_to_thread_result_and_context():
    value = contextvars.ContextVar("execution_context_value")

    async def run(executor):
        value.set(7)
        context = IRExecutionContext(executor)
        result = await context.to_thread(lambda x, *, y: value.get() + x + y, 2, y=3)
        assert result == 12

    with ThreadPoolExecutor(max_workers=1) as executor:
        asyncio.run(run(executor))


def test_to_thread_worker_exception():
    def worker():
        raise ValueError("worker failed")

    async def run(executor):
        context = IRExecutionContext(executor)
        with pytest.raises(ValueError, match="worker failed"):
            await context.to_thread(worker)

    with ThreadPoolExecutor(max_workers=1) as executor:
        asyncio.run(run(executor))
