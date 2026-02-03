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

        Parameters
        ----------
        table
            The table to record.
        """
        if table is not None:  # pragma: no cover; Covered by rapidsmpf tests
            self.row_count = (self.row_count or 0) + table.num_rows()
        self.chunk_count += 1

    def set_duplicated(self, *, duplicated: bool = True) -> None:
        """Mark output rows as duplicated across ranks."""
        self.duplicated = duplicated
