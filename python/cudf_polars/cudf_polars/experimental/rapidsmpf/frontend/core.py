# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Multi-GPU frontend core."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Self

import polars as pl

if TYPE_CHECKING:
    import contextlib


class StreamingEngine(pl.GPUEngine):
    """
    Base class for multi-GPU Polars engines backed by a streaming executor.

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
    exit_stack
        A :class:`contextlib.ExitStack` whose registered contexts are closed
        when :meth:`shutdown` is called.
    executor_options
        Key/value options forwarded to the streaming executor.
    engine_options
        Additional keyword arguments forwarded to
        :class:`~polars.lazyframe.engine_config.GPUEngine`.
    """

    def __init__(
        self,
        *,
        nranks: int,
        exit_stack: contextlib.ExitStack,
        executor_options: dict[str, object],
        engine_options: dict[str, Any],
    ):
        self._nranks = nranks
        self._exit_stack: contextlib.ExitStack | None = exit_stack
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
