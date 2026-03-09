# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Core implementation for trace-to-explain conversion."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

__all__ = [
    "NodeStats",
    "QueryPlan",
    "get_traces_for_query",
    "load_jsonl",
    "main",
]


@dataclass
class NodeStats:
    """Aggregated statistics for a single IR node (across all workers)."""

    ir_type: str = ""
    # From Streaming Actor events (aggregated across workers)
    chunk_count: int = 0
    rows: int | None = None
    duplicated: bool = False
    decision: str | None = None
    worker_count: int = 0  # Number of workers that reported this node
    # From Execute IR events (aggregated across chunks and workers)
    total_bytes_input: int = 0
    total_bytes_output: int = 0
    total_duration_ns: int = 0
    exec_count: int = 0  # Number of Execute IR events

    def add_streaming_actor(self, event: dict[str, Any]) -> None:
        """Add stats from a Streaming Actor event (one per worker)."""
        self.ir_type = event.get("actor_ir_type", self.ir_type)
        # duplicated should be consistent across workers
        is_duplicated = event.get("duplicated", False)
        self.duplicated = self.duplicated or is_duplicated

        rows = event.get("rows")
        if is_duplicated:
            # For duplicated nodes, data is replicated across workers,
            # so take max instead of sum (all workers have same data)
            self.chunk_count = max(self.chunk_count, event.get("chunk_count", 0))
            if rows is not None:
                self.rows = max(self.rows or 0, int(rows))
        else:
            # For partitioned nodes, aggregate across workers
            self.chunk_count += event.get("chunk_count", 0)
            if rows is not None:
                self.rows = (self.rows or 0) + int(rows)

        if event.get("decision"):
            self.decision = event.get("decision")
        self.worker_count += 1

    def add_execute_ir(self, event: dict[str, Any]) -> None:
        """Add stats from an Execute IR event."""
        self.ir_type = event.get("type", self.ir_type)
        self.total_bytes_input += event.get("total_bytes_input", 0)
        self.total_bytes_output += event.get("total_bytes_output", 0)
        start = event.get("start", 0)
        stop = event.get("stop", 0)
        if start and stop:
            self.total_duration_ns += stop - start
        self.exec_count += 1


@dataclass
class QueryPlan:
    """A query plan with tree structure and execution stats."""

    # ir_id can be int or str (MD5 hash) depending on the logging version
    nodes: dict[str | int, dict[str, Any]] = field(
        default_factory=dict
    )  # ir_id -> node info
    stats: dict[str | int, NodeStats] = field(
        default_factory=lambda: defaultdict(NodeStats)
    )
    root_id: str | int | None = None

    @classmethod
    def from_traces(cls, traces: list[dict[str, Any]]) -> QueryPlan:
        """Build a QueryPlan from a list of trace events."""
        plan = cls()

        for event in traces:
            scope = event.get("scope")
            event_type = event.get("event")

            if scope == "plan":
                # Query Plan / IR Structure event
                # Handle both old format (nodes list) and new format (SerializablePlan)
                if "plan" in event:
                    # New SerializablePlan format
                    ser_plan = event["plan"]
                    nodes_dict = ser_plan.get("nodes", {})
                    for ir_id, node in nodes_dict.items():
                        # Convert new format to internal format
                        plan.nodes[ir_id] = {
                            "ir_id": ir_id,
                            "ir_type": node.get("type", ""),
                            "children_ir_ids": node.get("children", []),
                            "schema": node.get("schema", {}),
                            "properties": node.get("properties", {}),
                        }
                        plan.stats[ir_id].ir_type = node.get("type", "")
                    # Root is in the roots list
                    roots = ser_plan.get("roots", [])
                    if roots:
                        plan.root_id = roots[0]
                else:
                    # Old format (list of nodes)
                    nodes_list = event.get("nodes", [])
                    for node in nodes_list:
                        ir_id = node["ir_id"]
                        plan.nodes[ir_id] = node
                        plan.stats[ir_id].ir_type = node.get("ir_type", "")
                    # First node is the root
                    if nodes_list:
                        plan.root_id = nodes_list[0]["ir_id"]

            elif scope == "actor" or event_type == "Streaming Actor":
                # Streaming Actor event (one per worker per IR node)
                ir_id = event.get("actor_ir_id")
                if ir_id is not None:
                    # Convert to string for consistency with new format
                    ir_id = str(ir_id) if isinstance(ir_id, int) else ir_id
                    plan.stats[ir_id].add_streaming_actor(event)

            elif scope == "evaluate_ir_node" or event_type == "Execute IR":
                # Execute IR event (detailed per-chunk execution stats)
                # These events have actor_ir_id bound via contextvars
                ir_id = event.get("actor_ir_id") or event.get("ir_id")
                if ir_id is not None:
                    # Convert to string for consistency with new format
                    ir_id = str(ir_id) if isinstance(ir_id, int) else ir_id
                    plan.stats[ir_id].add_execute_ir(event)

        return plan

    def render(self) -> str:
        """Render the query plan as a tree string."""
        if self.root_id is None:
            # No tree structure, show flat stats
            return self._render_flat()

        lines: list[str] = []
        self._render_node(self.root_id, "", lines)
        return "\n".join(lines)

    def _render_flat(self) -> str:
        """Render stats as a flat list when no tree structure is available."""
        if not self.stats:
            return "(no trace data found)"

        lines = ["(no query plan tree - showing flat stats by ir_id)"]
        # Group by ir_type for readability
        by_type: dict[str, list[tuple[str | int, NodeStats]]] = defaultdict(list)
        for ir_id, stats in self.stats.items():
            by_type[stats.ir_type or "Unknown"].append((ir_id, stats))

        for ir_type in sorted(by_type.keys()):
            items = by_type[ir_type]
            # Aggregate stats for this type
            total_rows = sum(s.rows or 0 for _, s in items)
            total_chunks = sum(s.chunk_count for _, s in items)
            total_bytes = sum(s.total_bytes_output for _, s in items)
            total_time = sum(s.total_duration_ns for _, s in items)
            workers = max((s.worker_count for _, s in items), default=0)

            annotations = []
            if total_rows > 0:
                annotations.append(f"rows={_fmt_count(total_rows)}")
            if total_chunks > 0:
                annotations.append(f"chunks={total_chunks}")
            if total_bytes > 0:
                annotations.append(f"bytes_out={_fmt_bytes(total_bytes)}")
            if total_time > 0:
                annotations.append(f"time={_fmt_duration(total_time)}")
            if workers > 1:
                annotations.append(f"workers={workers}")

            annotation_str = f" [{', '.join(annotations)}]" if annotations else ""
            lines.append(f"  {ir_type.upper()}{annotation_str}")

        return "\n".join(lines)

    def _render_node(self, ir_id: str | int, indent: str, lines: list[str]) -> None:
        """Recursively render a node and its children."""
        node = self.nodes.get(ir_id, {})
        stats = self.stats.get(ir_id, NodeStats())

        ir_type = stats.ir_type or node.get("ir_type", "Unknown")

        # Build the annotation string
        annotations = []

        # Row count
        if stats.rows is not None:
            annotations.append(f"rows={_fmt_count(stats.rows)}")

        # Chunk count
        if stats.chunk_count > 0:
            annotations.append(f"chunks={stats.chunk_count}")

        # Bytes output
        if stats.total_bytes_output > 0:
            annotations.append(f"bytes_out={_fmt_bytes(stats.total_bytes_output)}")

        # Duration
        if stats.total_duration_ns > 0:
            annotations.append(f"time={_fmt_duration(stats.total_duration_ns)}")

        # Decision (for joins, groupbys, etc.)
        if stats.decision:
            annotations.append(f"decision={stats.decision}")

        # Worker count (for multi-rank)
        if stats.worker_count > 1:
            annotations.append(f"workers={stats.worker_count}")

        # Duplicated flag
        if stats.duplicated:
            annotations.append("duplicated")

        # Add properties from the plan node (keys, is_pointwise, etc.)
        properties = node.get("properties", {})

        # Show keys for GROUPBY
        if ir_type == "GroupBy" and "keys" in properties:
            keys = properties["keys"]
            if keys:
                annotations.append(f"keys={tuple(keys)}")

        # Show keys for JOIN
        if ir_type == "Join" and "left_on" in properties:
            left_on = properties.get("left_on", [])
            if left_on:
                annotations.append(f"on={tuple(left_on)}")

        # Format the line
        annotation_str = f" [{', '.join(annotations)}]" if annotations else ""
        lines.append(f"{indent}{ir_type.upper()}{annotation_str}")

        # Render children
        children_ids = node.get("children_ir_ids", [])
        for child_id in children_ids:
            self._render_node(child_id, indent + "  ", lines)


def _fmt_count(value: int) -> str:
    """Format a count as a readable string."""
    if value < 1_000:
        return str(value)
    elif value < 1_000_000:
        return f"{value / 1_000:.2g}K"
    elif value < 1_000_000_000:
        return f"{value / 1_000_000:.2g}M"
    else:
        return f"{value / 1_000_000_000:.2g}B"


def _fmt_bytes(value: int) -> str:
    """Format bytes as a readable string."""
    if value < 1024:
        return f"{value}B"
    elif value < 1024 * 1024:
        return f"{value / 1024:.2g}KB"
    elif value < 1024 * 1024 * 1024:
        return f"{value / (1024 * 1024):.2g}MB"
    else:
        return f"{value / (1024 * 1024 * 1024):.2g}GB"


def _fmt_duration(ns: int) -> str:
    """Format nanoseconds as a readable duration string."""
    if ns < 1_000:
        return f"{ns}ns"
    elif ns < 1_000_000:
        return f"{ns / 1_000:.2g}us"
    elif ns < 1_000_000_000:
        return f"{ns / 1_000_000:.2g}ms"
    else:
        return f"{ns / 1_000_000_000:.2g}s"


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Load all records from a JSONL file."""
    records = []
    with Path(path).open() as f:
        for raw_line in f:
            line = raw_line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _has_query_plan(traces: list[dict[str, Any]]) -> bool:
    """Check if traces contain a Query Plan event with node structure."""
    for t in traces:
        if t.get("scope") == "plan":
            # Check for new format (SerializablePlan with plan dict)
            if "plan" in t and t["plan"].get("nodes"):
                return True
            # Check for old format (nodes list)
            if t.get("nodes"):
                return True
    return False


def get_traces_for_query(
    records: list[dict[str, Any]],
    query_id: int | None = None,
    iteration: int = 0,
) -> tuple[int, list[dict[str, Any]]] | None:
    """
    Extract traces for a specific query and iteration.

    If query_id is None, returns the first query found with traces.
    Prioritizes traces that include a Query Plan event.
    """
    # First pass: look for traces with Query Plan
    for record in records:
        query_records = record.get("records", {})
        for qid_str, iterations in query_records.items():
            qid = int(qid_str)
            if query_id is not None and qid != query_id:
                continue
            if iteration < len(iterations):
                traces = iterations[iteration].get("traces")
                if traces and _has_query_plan(traces):
                    return qid, traces

    # Second pass: fall back to any traces
    for record in records:
        query_records = record.get("records", {})
        for qid_str, iterations in query_records.items():
            qid = int(qid_str)
            if query_id is not None and qid != query_id:
                continue
            if iteration < len(iterations):
                traces = iterations[iteration].get("traces")
                if traces:
                    return qid, traces
    return None


def main() -> None:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(
        description="Convert benchmark traces to explain-like output."
    )
    parser.add_argument(
        "input",
        type=Path,
        help="Path to the JSONL benchmark results file.",
    )
    parser.add_argument(
        "--query",
        "-q",
        type=int,
        default=None,
        help="Query ID to display. If not specified, shows the first query with traces.",
    )
    parser.add_argument(
        "--iteration",
        "-i",
        type=int,
        default=0,
        help="Iteration number (default: 0).",
    )
    parser.add_argument(
        "--list",
        "-l",
        action="store_true",
        help="List all available queries with traces and exit.",
    )
    parser.add_argument(
        "--all",
        "-a",
        action="store_true",
        help="Show all queries with traces.",
    )
    args = parser.parse_args()

    if not args.input.exists():
        print(f"Error: File not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    records = load_jsonl(args.input)

    seen: set[tuple[int, int]] = set()

    if args.list:
        # List available queries
        print("Available queries with traces:")
        for record in records:
            query_records = record.get("records", {})
            for qid_str, iterations in query_records.items():
                qid = int(qid_str)
                for i, it in enumerate(iterations):
                    traces = it.get("traces")
                    if traces and (qid, i) not in seen:
                        seen.add((qid, i))
                        duration = it.get("duration", 0)
                        has_plan = " [has plan]" if _has_query_plan(traces) else ""
                        print(
                            f"  Query {qid}, Iteration {i}: {duration:.3f}s{has_plan}"
                        )
        return

    if args.all:
        # Show all queries
        for record in records:
            query_records = record.get("records", {})
            for qid_str, iterations in query_records.items():
                qid = int(qid_str)
                for i, it in enumerate(iterations):
                    traces = it.get("traces")
                    if traces and (qid, i) not in seen:
                        seen.add((qid, i))
                        duration = it.get("duration", 0)
                        print(f"\n{'=' * 60}")
                        print(f"Query {qid}, Iteration {i} (duration: {duration:.3f}s)")
                        print("=" * 60)
                        plan = QueryPlan.from_traces(traces)
                        print(plan.render())
        return

    # Get traces for specific query
    result = get_traces_for_query(records, args.query, args.iteration)
    if result is None:
        if args.query is not None:
            print(
                f"Error: No traces found for query {args.query}, "
                f"iteration {args.iteration}",
                file=sys.stderr,
            )
        else:
            print("Error: No traces found in the file", file=sys.stderr)
        sys.exit(1)

    query_id, traces = result
    plan = QueryPlan.from_traces(traces)

    print(f"Query {query_id}, Iteration {args.iteration}")
    print("=" * 40)
    print(plan.render())
