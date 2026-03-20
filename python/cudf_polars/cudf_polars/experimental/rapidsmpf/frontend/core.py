# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Multi-GPU frontend core."""

from __future__ import annotations

from typing import Any, Self

import polars as pl


class StreamingEngine(pl.GPUEngine):
    """Base class for multi-GPU Polars engines backed by a streaming executor."""

    def __init__(
        self,
        *,
        nranks: int,
        executor_options: dict[str, object],
        engine_options: dict[str, Any],
    ):
        self._nranks = nranks
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
        Shut down engine.

        Must be called on the same thread that created the engine initially.
        """
        self.device = None
        self.memory_resource = None
        self.config = {}

    def __enter__(self) -> Self:
        """Enter the context manager, returning ``self``."""
        return self

    def __exit__(self, *_: object) -> None:
        """Exit the context manager, calling :meth:`shutdown`."""
        self.shutdown()
