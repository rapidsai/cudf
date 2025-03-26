# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Timing utilities."""

from __future__ import annotations

__all__: list[str] = ["Timer"]


class Timer:
    """
    A timer for recording execution times of nodes.

    Parameters
    ----------
    query_start
        Duration in nanoseconds since the query was started on the
        Polars side
    """

    def __init__(self, query_start: int):
        self.query_start = query_start
        self.timings: list[tuple[int, int, str]] = []

    def store(self, start: int, end: int, name: str) -> None:
        """
        Store timing for a node.

        Parameters
        ----------
        start
            Start of the execution for this node (use time.monotonic_ns).
        end
            End of the execution for this node.
        name
            The name for this node.
        """
        self.timings.append((start - self.query_start, end - self.query_start, name))
