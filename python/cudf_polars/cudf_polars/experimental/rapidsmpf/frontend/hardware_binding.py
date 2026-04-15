# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Topology-aware hardware binding."""

from __future__ import annotations

import dataclasses
import threading

from rapidsmpf import bootstrap
from rapidsmpf.rrun.rrun import bind


@dataclasses.dataclass(frozen=True)
class HardwareBindingPolicy:
    """
    Policy controlling topology-aware hardware binding.

    Determines whether :func:`rapidsmpf.rrun.rrun.bind` is invoked
    to pin the calling process to CPU cores, NUMA memory nodes,
    and network devices local to the worker's GPU.

    The GPU to bind to is resolved from ``CUDA_VISIBLE_DEVICES``.
    Each frontend is responsible for setting this variable per worker
    (Dask via the nanny preload or ``SpecCluster``, Ray via
    ``num_gpus=1`` scheduling, SPMD via ``rrun``). If
    ``CUDA_VISIBLE_DEVICES`` is unset, binding falls back to GPU 0.

    The default instance (``HardwareBindingPolicy()``) enables
    binding once per process with soft failure handling.

    Parameters
    ----------
    skip_under_rrun
        When ``True`` (the default), binding is skipped if the process
        was launched via ``rrun``, because ``rrun`` already performs
        hardware binding at launch time. If binding is skipped, all
        other options are ignored.
    enabled
        Whether binding is enabled. ``False`` disables all binding.
    enable_once
        When ``True``, binding is performed at most once per process;
        subsequent calls to :func:`bind_to_gpu` are no-ops. When
        ``False``, binding is attempted on every call.
    raise_on_fail
        When ``True``, binding failures raise an exception. Currently
        ``rrun.bind()`` only prints warnings to stderr on per-subsystem
        failures; setting this flag enables ``verbose=True`` so those
        warnings are visible.
    """

    skip_under_rrun: bool = True
    enabled: bool = True
    enable_once: bool = True
    raise_on_fail: bool = False


_bind_lock = threading.Lock()
_bind_done = False


def bind_to_gpu(policy: HardwareBindingPolicy) -> None:
    """
    Bind the calling process to resources topologically close to a GPU.

    Thread-safe wrapper around :func:`rapidsmpf.rrun.rrun.bind` governed
    by *policy*.

    When ``policy.enable_once`` is ``True`` (the default), double-checked
    locking with a module-level lock guarantees that the underlying
    ``bind()`` is called **at most once per process**, regardless of how
    many frontend engines are constructed or from how many threads.

    Parameters
    ----------
    policy
        The :class:`HardwareBindingPolicy` controlling binding behavior.
    """
    global _bind_done  # noqa: PLW0603
    if not policy.enabled:
        return
    if policy.skip_under_rrun and bootstrap.is_running_with_rrun():
        return
    if policy.enable_once:
        if _bind_done:
            return
        with _bind_lock:
            if _bind_done:
                return
            _do_bind(policy)
            _bind_done = True
    else:
        _do_bind(policy)


def _do_bind(policy: HardwareBindingPolicy) -> None:
    """Execute the actual bind call according to *policy*."""
    verbose = policy.raise_on_fail
    # TODO(rapidsmpf): rrun.bind() does not currently report per-subsystem
    # failures programmatically — it only prints warnings to stderr when
    # verbose=True. When a return value or exception is available, check it
    # here and raise if policy.raise_on_fail is True.
    # See: https://github.com/rapidsai/rapidsmpf/issues/963
    try:
        bind(verbose=verbose)
    except RuntimeError:
        # CUDA_VISIBLE_DEVICES is unset; fall back to GPU 0.
        bind(gpu_id=0, verbose=verbose)
