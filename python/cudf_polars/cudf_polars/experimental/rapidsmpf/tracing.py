# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Tracing infrastructure for the RapidsMPF streaming runtime."""

from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pylibcudf as plc

    from cudf_polars.dsl.ir import IR


def _stable_ir_id(ir_node: IR) -> int:
    """
    Compute a stable identifier for an IR node.

    Uses MD5 hash of the node's hashable representation for determinism
    across process boundaries (Python's hash() uses PYTHONHASHSEED).

    Parameters
    ----------
    ir_node
        The IR node.

    Returns
    -------
    int
        A stable 32-bit identifier for this node.
    """
    content = repr(ir_node.get_hashable()).encode("utf-8")
    return int(hashlib.md5(content).hexdigest()[:8], 16)


class StreamingNodeTracer:
    """
    Tracer for a single streaming IR node.

    Collects execution statistics and emits structured log events.

    Attributes
    ----------
    ir_id
        Stable identifier for the IR node (for tracing/logging).
    ir_type
        Type name of the IR node (e.g., "Sort", "Join").
    row_count
        Total row count produced by this node during execution.
        None if row counting is not available for this node.
    chunk_count
        Total chunk count produced by this node during execution.
    decision
        The algorithm decision made at runtime for this node
        (e.g., "broadcast_left", "shuffle", "tree", etc.).
    duplicated
        Whether the output rows are duplicated across ranks
        (e.g., after an allgather). Affects how rows are merged.
    """

    __slots__ = (
        "chunk_count",
        "decision",
        "duplicated",
        "ir_id",
        "ir_type",
        "row_count",
    )

    def __init__(self, ir_id: int | None = None, ir_type: str | None = None) -> None:
        self.ir_id = ir_id
        self.ir_type = ir_type
        self.row_count: int | None = None
        self.chunk_count: int = 0
        self.decision: str | None = None
        self.duplicated: bool = False

    def add_chunk(self, *, table: plc.Table | None = None) -> None:
        """
        Record a chunk.

        If table is provided, both row_count and chunk_count are updated.
        If table is None, only chunk_count is incremented.
        """
        if table is not None:  # pragma: no cover; Covered by rapidsmpf tests
            self.row_count = (self.row_count or 0) + table.num_rows()
        self.chunk_count += 1

    def set_duplicated(self, *, duplicated: bool = True) -> None:
        """
        Mark output rows as duplicated across ranks.

        Call this after sending metadata when the output is duplicated
        (e.g., after an allgather). Affects how rows are merged across ranks.
        """
        self.duplicated = duplicated

    def merge(self, other: StreamingNodeTracer) -> None:
        """Merge another node tracer's stats into this one."""
        if other.row_count is not None:
            if self.duplicated or other.duplicated:
                # For duplicated data, take max (don't sum across ranks)
                self.row_count = max(self.row_count or 0, other.row_count)
                self.duplicated = True
            else:
                self.row_count = (self.row_count or 0) + other.row_count
        self.chunk_count += other.chunk_count
        if other.decision is not None:
            self.decision = other.decision


class StreamingQueryTracer:
    """
    Tracer for collecting runtime statistics for an entire streaming query.

    Attributes
    ----------
    node_tracers
        Mapping from each IR node to its node tracer.
    """

    __slots__ = ("node_tracers",)
    node_tracers: dict[IR, StreamingNodeTracer]

    def __init__(self) -> None:
        self.node_tracers = {}

    def get_or_create(self, ir: IR) -> StreamingNodeTracer:
        """
        Get or create a node tracer for the given IR.

        Use this when setting up tracing for a node. To check if a node
        was traced without creating an entry, use `node_tracers.get(ir)`.
        """
        if ir not in self.node_tracers:
            ir_id = _stable_ir_id(ir)
            ir_type = type(ir).__name__
            self.node_tracers[ir] = StreamingNodeTracer(ir_id, ir_type)
        return self.node_tracers[ir]

    def merge(self, other: StreamingQueryTracer) -> None:
        """Merge another query tracer's statistics into this one."""
        for ir, node_tracer in other.node_tracers.items():
            self.get_or_create(ir).merge(node_tracer)
