# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Tests for bind_to_gpu() and HardwareBindingPolicy."""

from __future__ import annotations

import multiprocessing
import traceback
from typing import TYPE_CHECKING
from unittest.mock import MagicMock, call, patch

import distributed
import pytest

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
    try:
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
    finally:
        proc.close()


def _reset_bind_state() -> None:
    """Reset the module-level bind state so each subprocess starts clean."""
    from cudf_polars.experimental.rapidsmpf.frontend import hardware_binding

    hardware_binding._bind_done = False


def test_bind_called_once() -> None:
    """bind() is called exactly once even when bind_to_gpu() is called twice."""

    def body() -> None:
        _reset_bind_state()
        with patch(
            "cudf_polars.experimental.rapidsmpf.frontend.hardware_binding.bind"
        ) as mock_bind:
            from cudf_polars.experimental.rapidsmpf.frontend.hardware_binding import (
                HardwareBindingPolicy,
                bind_to_gpu,
            )

            policy = HardwareBindingPolicy()
            bind_to_gpu(policy)
            bind_to_gpu(policy)
            assert mock_bind.call_count == 1

    _run_in_subprocess(body)


def test_bind_falls_back_to_gpu_0() -> None:
    """When bind() raises RuntimeError, falls back to gpu_id=0."""

    def body() -> None:
        _reset_bind_state()
        mock_bind = MagicMock(
            side_effect=[RuntimeError("no CUDA_VISIBLE_DEVICES"), None]
        )
        with patch(
            "cudf_polars.experimental.rapidsmpf.frontend.hardware_binding.bind",
            mock_bind,
        ):
            from cudf_polars.experimental.rapidsmpf.frontend.hardware_binding import (
                HardwareBindingPolicy,
                bind_to_gpu,
            )

            policy = HardwareBindingPolicy()
            bind_to_gpu(policy)
            bind_kw = {
                "cpu": policy.cpu,
                "memory": policy.memory,
                "network": policy.network,
            }
            assert mock_bind.call_count == 2
            assert mock_bind.call_args_list == [
                call(**bind_kw),
                call(gpu_id=0, **bind_kw),
            ]

    _run_in_subprocess(body)


def test_bind_raise_on_fail_propagates_exception() -> None:
    """raise_on_fail=True lets RuntimeError from bind() propagate."""

    def body() -> None:
        _reset_bind_state()
        mock_bind = MagicMock(side_effect=RuntimeError("binding failed"))
        with patch(
            "cudf_polars.experimental.rapidsmpf.frontend.hardware_binding.bind",
            mock_bind,
        ):
            from cudf_polars.experimental.rapidsmpf.frontend.hardware_binding import (
                HardwareBindingPolicy,
                bind_to_gpu,
            )

            with pytest.raises(RuntimeError, match="binding failed"):
                bind_to_gpu(HardwareBindingPolicy(raise_on_fail=True))

    _run_in_subprocess(body)


def test_bind_raise_on_fail_false_suppresses_exception() -> None:
    """raise_on_fail=False silently ignores RuntimeError from bind()."""

    def body() -> None:
        _reset_bind_state()
        mock_bind = MagicMock(side_effect=RuntimeError("binding failed"))
        with patch(
            "cudf_polars.experimental.rapidsmpf.frontend.hardware_binding.bind",
            mock_bind,
        ):
            from cudf_polars.experimental.rapidsmpf.frontend.hardware_binding import (
                HardwareBindingPolicy,
                bind_to_gpu,
            )

            bind_to_gpu(HardwareBindingPolicy(raise_on_fail=False))

    _run_in_subprocess(body)


def test_bind_thread_safe() -> None:
    """Concurrent calls from multiple threads result in exactly one bind() call."""

    def body() -> None:
        import threading

        _reset_bind_state()
        with patch(
            "cudf_polars.experimental.rapidsmpf.frontend.hardware_binding.bind"
        ) as mock_bind:
            from cudf_polars.experimental.rapidsmpf.frontend.hardware_binding import (
                HardwareBindingPolicy,
                bind_to_gpu,
            )

            policy = HardwareBindingPolicy()
            barrier = threading.Barrier(8)

            def _call_bind() -> None:
                barrier.wait()
                bind_to_gpu(policy)

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
        from cudf_polars.experimental.rapidsmpf.frontend import hardware_binding

        _reset_bind_state()
        assert not hardware_binding._bind_done
        with patch("cudf_polars.experimental.rapidsmpf.frontend.hardware_binding.bind"):
            hardware_binding.bind_to_gpu(hardware_binding.HardwareBindingPolicy())
            assert hardware_binding._bind_done

    _run_in_subprocess(body)


def test_bind_disabled() -> None:
    """enabled=False skips binding entirely."""

    def body() -> None:
        _reset_bind_state()
        with patch(
            "cudf_polars.experimental.rapidsmpf.frontend.hardware_binding.bind"
        ) as mock_bind:
            from cudf_polars.experimental.rapidsmpf.frontend.hardware_binding import (
                HardwareBindingPolicy,
                bind_to_gpu,
            )

            bind_to_gpu(HardwareBindingPolicy(enabled=False))
            mock_bind.assert_not_called()

    _run_in_subprocess(body)


def test_bind_enable_once_false() -> None:
    """enable_once=False allows repeated bind() calls."""

    def body() -> None:
        _reset_bind_state()
        with patch(
            "cudf_polars.experimental.rapidsmpf.frontend.hardware_binding.bind"
        ) as mock_bind:
            from cudf_polars.experimental.rapidsmpf.frontend.hardware_binding import (
                HardwareBindingPolicy,
                bind_to_gpu,
            )

            policy = HardwareBindingPolicy(enable_once=False)
            bind_to_gpu(policy)
            bind_to_gpu(policy)
            assert mock_bind.call_count == 2

    _run_in_subprocess(body)


# ---------------------------------------------------------------------------
# dask_setup (nanny preload)
# ---------------------------------------------------------------------------


def test_get_visible_gpu_ids_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    from cudf_polars.experimental.rapidsmpf.frontend.dask import _get_visible_gpu_ids

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "3,1,4")
    assert _get_visible_gpu_ids() == ["3", "1", "4"]


def test_dask_setup_assigns_gpus(monkeypatch: pytest.MonkeyPatch) -> None:
    """dask_setup assigns round-robin CUDA_VISIBLE_DEVICES to each nanny."""
    import cudf_polars.experimental.rapidsmpf.frontend.dask as dask_mod

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1")
    monkeypatch.setattr(dask_mod, "_nanny_preload_counter", 0)

    nanny0 = MagicMock(spec=distributed.Nanny)
    nanny0.env = {}
    dask_mod.dask_setup(nanny0)
    assert nanny0.env["CUDA_VISIBLE_DEVICES"] == "0"

    nanny1 = MagicMock(spec=distributed.Nanny)
    nanny1.env = {}
    dask_mod.dask_setup(nanny1)
    assert nanny1.env["CUDA_VISIBLE_DEVICES"] == "1"


def test_dask_setup_wraps_around(monkeypatch: pytest.MonkeyPatch) -> None:
    """Counter wraps around when workers exceed GPUs."""
    import cudf_polars.experimental.rapidsmpf.frontend.dask as dask_mod

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1")
    monkeypatch.setattr(dask_mod, "_nanny_preload_counter", 0)

    nannies = []
    for _ in range(4):
        n = MagicMock(spec=distributed.Nanny)
        n.env = {}
        dask_mod.dask_setup(n)
        nannies.append(n)

    assert nannies[0].env["CUDA_VISIBLE_DEVICES"] == "0"
    assert nannies[1].env["CUDA_VISIBLE_DEVICES"] == "1"
    assert nannies[2].env["CUDA_VISIBLE_DEVICES"] == "0"  # wraps
    assert nannies[3].env["CUDA_VISIBLE_DEVICES"] == "1"  # wraps


def test_dask_setup_rejects_worker() -> None:
    """dask_setup raises TypeError when used with --preload instead of --preload-nanny."""
    import cudf_polars.experimental.rapidsmpf.frontend.dask as dask_mod

    worker = MagicMock(spec=distributed.Worker)
    with pytest.raises(TypeError, match="--preload-nanny"):
        dask_mod.dask_setup(worker)
