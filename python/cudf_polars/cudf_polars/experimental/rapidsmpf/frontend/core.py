# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Multi-GPU frontend core."""

from __future__ import annotations

import contextlib
from typing import Any, Self

import polars as pl

_RESERVED_EXECUTOR_KEYS: frozenset[str] = frozenset(
    {"runtime", "cluster", "spmd_context", "ray_context", "dask_context"}
)
_RESERVED_ENGINE_KEYS: frozenset[str] = frozenset({"memory_resource", "executor"})


def check_reserved_keys(
    executor_options: dict[str, Any],
    engine_options: dict[str, Any],
) -> None:
    """
    Raise :exc:`TypeError` if any reserved keys are present in the option dicts.

    Parameters
    ----------
    executor_options
        Executor-specific options to validate.
    engine_options
        Engine-specific options to validate.

    Raises
    ------
    TypeError
        If ``executor_options`` contains any reserved key.
    TypeError
        If ``engine_options`` contains any reserved key.
    """
    if bad := _RESERVED_EXECUTOR_KEYS & executor_options.keys():
        raise TypeError(f"executor_options may not contain reserved keys: {bad}")
    if bad := _RESERVED_ENGINE_KEYS & engine_options.keys():
        raise TypeError(f"engine_options may not contain reserved keys: {bad}")


class StreamingEngine(pl.GPUEngine):
    """
    Base class for multi-GPU Polars engines.

    The engine manages the lifecycle of a streaming execution and can
    be used as a context manager. On exit, :meth:`shutdown` is called.

    Notes
    -----
    The engine must be created and shut down on the same thread. In particular,
    destruction and context manager exit must occur on the thread that created
    the instance.

    Parameters
    ----------
    nranks
        Number of ranks (workers or GPUs) in the cluster.
    executor_options
        Executor-specific options (e.g. ``max_rows_per_partition``).
    engine_options
        Engine-specific keyword arguments (e.g. ``raise_on_fail``,
        ``parquet_options``).
    exit_stack
        A :class:`contextlib.ExitStack` whose registered contexts are closed
        when :meth:`shutdown` is called. If ``None``, an empty stack is created.
    """

    def __init__(
        self,
        *,
        nranks: int,
        executor_options: dict[str, Any],
        engine_options: dict[str, Any],
        exit_stack: contextlib.ExitStack | None = None,
    ):
        self._nranks = nranks
        self._exit_stack: contextlib.ExitStack | None = (
            exit_stack or contextlib.ExitStack()
        )
        super().__init__(
            executor="streaming",
            executor_options=executor_options,
            **engine_options,
        )

    @property
    def nranks(self) -> int:
        """
        Number of ranks (for example GPUs or workers) in the cluster.

        Local execution without a cluster returns 1.

        Returns
        -------
        Number of ranks.
        """
        return self._nranks

    def shutdown(self) -> None:
        """
        Shut down engine and release all owned resources.

        Idempotent: safe to call more than once. Must be called on the same
        thread that created the engine.
        """
        if self._exit_stack is None:
            return  # already shut down
        try:
            self._exit_stack.close()
        finally:
            self._exit_stack = None
            self.device = None
            self.memory_resource = None
            self.config = {}

    def __enter__(self) -> Self:
        """Enter the context manager, returning ``self``."""
        return self

    def __exit__(self, *_: object) -> None:
        """Exit the context manager, calling :meth:`shutdown`."""
        self.shutdown()
