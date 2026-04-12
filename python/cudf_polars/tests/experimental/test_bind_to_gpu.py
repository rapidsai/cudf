# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Tests for bind_to_gpu() thread-safe wrapper."""

from __future__ import annotations

import multiprocessing
import traceback
from typing import TYPE_CHECKING
from unittest.mock import MagicMock, call, patch

if TYPE_CHECKING:
    from collections.abc import Callable


def _run_in_subprocess(target: Callable[[], None]) -> None:
    """Execute ``target()`` in a forked child process.

    Because each call forks a new child, process-wide side-effects
    (the ``_bind_done`` flag, CPU affinity, environment variables) never
    leak between tests or back into the pytest process.
    """
    ctx = multiprocessing.get_context("fork")
    parent_conn, child_conn = ctx.Pipe()

    def _wrapper() -> None:
        try:
            target()
            child_conn.send(None)
        except BaseException as exc:
            try:
                child_conn.send(exc)
            except Exception:
                child_conn.send(
                    RuntimeError(
                        f"{type(exc).__name__}: {exc}\n"
                        f"{''.join(traceback.format_tb(exc.__traceback__))}"
                    )
                )
        finally:
            child_conn.close()

    proc = ctx.Process(target=_wrapper)
    proc.start()
    proc.join(timeout=30)

    if proc.is_alive():
        proc.kill()
        proc.join()
        raise RuntimeError("Subprocess timed out after 30 seconds")

    if parent_conn.poll():
        exc = parent_conn.recv()
        if exc is not None:
            raise exc

    if proc.exitcode != 0:
        raise RuntimeError(f"Subprocess exited with code {proc.exitcode}")


def _reset_bind_state() -> None:
    """Reset the module-level bind state so each subprocess starts clean."""
    from cudf_polars.experimental.rapidsmpf.frontend import core

    core._bind_done = False


def test_bind_called_once() -> None:
    """bind() is called exactly once even when bind_to_gpu() is called twice."""

    def body() -> None:
        _reset_bind_state()
        with patch(
            "cudf_polars.experimental.rapidsmpf.frontend.core.bind"
        ) as mock_bind:
            from cudf_polars.experimental.rapidsmpf.frontend.core import (
                bind_to_gpu,
            )

            bind_to_gpu(verbose=False)
            bind_to_gpu(verbose=False)
            assert mock_bind.call_count == 1

    _run_in_subprocess(body)


def test_bind_falls_back_to_gpu_0() -> None:
    """When bind() raises RuntimeError, falls back to gpu_id=0."""

    def body() -> None:
        _reset_bind_state()
        mock_bind = MagicMock(
            side_effect=[RuntimeError("no CUDA_VISIBLE_DEVICES"), None]
        )
        with patch("cudf_polars.experimental.rapidsmpf.frontend.core.bind", mock_bind):
            from cudf_polars.experimental.rapidsmpf.frontend.core import (
                bind_to_gpu,
            )

            bind_to_gpu(verbose=False)
            assert mock_bind.call_count == 2
            assert mock_bind.call_args_list == [
                call(verbose=False),
                call(gpu_id=0, verbose=False),
            ]

    _run_in_subprocess(body)


def test_bind_forwards_verbose() -> None:
    """The verbose parameter is forwarded to bind()."""

    def body() -> None:
        _reset_bind_state()
        with patch(
            "cudf_polars.experimental.rapidsmpf.frontend.core.bind"
        ) as mock_bind:
            from cudf_polars.experimental.rapidsmpf.frontend.core import (
                bind_to_gpu,
            )

            bind_to_gpu(verbose=True)
            mock_bind.assert_called_once_with(verbose=True)

    _run_in_subprocess(body)


def test_bind_thread_safe() -> None:
    """Concurrent calls from multiple threads result in exactly one bind() call."""

    def body() -> None:
        import threading

        _reset_bind_state()
        with patch(
            "cudf_polars.experimental.rapidsmpf.frontend.core.bind"
        ) as mock_bind:
            from cudf_polars.experimental.rapidsmpf.frontend.core import (
                bind_to_gpu,
            )

            barrier = threading.Barrier(8)

            def _call_bind() -> None:
                barrier.wait()
                bind_to_gpu(verbose=False)

            threads = [threading.Thread(target=_call_bind) for _ in range(8)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()

            assert mock_bind.call_count == 1

    _run_in_subprocess(body)


def test_bind_done_flag_set() -> None:
    """_bind_done is True after bind_to_gpu() succeeds."""

    def body() -> None:
        from cudf_polars.experimental.rapidsmpf.frontend import core

        _reset_bind_state()
        assert not core._bind_done
        with patch("cudf_polars.experimental.rapidsmpf.frontend.core.bind"):
            core.bind_to_gpu(verbose=False)
            assert core._bind_done

    _run_in_subprocess(body)
