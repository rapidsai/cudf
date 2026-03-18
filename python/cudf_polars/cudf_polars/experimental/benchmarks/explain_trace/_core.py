# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Iterator


@dataclass
class NodeStats:
    ir_type: str = ""
    chunk_count: int = 0
    rows: int | None = None
    duplicated: bool = False
    decision: str | None = None
    worker_count: int = 0
    total_bytes_input: int = 0
    total_bytes_output: int = 0
    total_duration_ns: int = 0
    exec_count: int = 0

    def add_streaming_actor(self, event: dict[str, Any]) -> None:
        self.ir_type = event.get("actor_ir_type", self.ir_type)
        is_dup = event.get("duplicated", False)
        self.duplicated = self.duplicated or is_dup
        agg = max if is_dup else lambda a, b: a + b
        self.chunk_count = agg(self.chunk_count, event.get("chunk_count", 0))
        if (rows := event.get("rows")) is not None:
            self.rows = agg(self.rows or 0, int(rows))
        if event.get("decision"):
            self.decision = event["decision"]
        self.worker_count += 1

    def add_execute_ir(self, event: dict[str, Any]) -> None:
        self.ir_type = event.get("type", self.ir_type)
        self.total_bytes_input += event.get("total_bytes_input", 0)
        self.total_bytes_output += event.get("total_bytes_output", 0)
        if (start := event.get("start")) and (stop := event.get("stop")):
            self.total_duration_ns += stop - start
        self.exec_count += 1


@dataclass
class QueryPlan:
    nodes: dict[str | int, dict[str, Any]] = field(default_factory=dict)
    stats: dict[str | int, NodeStats] = field(
        default_factory=lambda: defaultdict(NodeStats)
    )
    root_id: str | int | None = None

    @classmethod
    def from_traces(cls, traces: list[dict[str, Any]]) -> QueryPlan:
        plan = cls()
        for event in traces:
            scope, etype = event.get("scope"), event.get("event")
            if scope == "plan":
                if "plan" in event:
                    ser = event["plan"]
                    for ir_id, node in ser.get("nodes", {}).items():
                        plan.nodes[ir_id] = {
                            "ir_id": ir_id,
                            "ir_type": node.get("type", ""),
                            "children_ir_ids": node.get("children", []),
                            "schema": node.get("schema", {}),
                            "properties": node.get("properties", {}),
                        }
                        plan.stats[ir_id].ir_type = node.get("type", "")
                    if roots := ser.get("roots"):
                        plan.root_id = roots[0]
                else:
                    for node in event.get("nodes", []):
                        ir_id = node["ir_id"]
                        plan.nodes[ir_id] = node
                        plan.stats[ir_id].ir_type = node.get("ir_type", "")
                    if nodes := event.get("nodes"):
                        plan.root_id = nodes[0]["ir_id"]
            elif scope == "actor" or etype == "Streaming Actor":
                if (ir_id := event.get("actor_ir_id")) is not None:
                    ir_id = str(ir_id) if isinstance(ir_id, int) else ir_id
                    plan.stats[ir_id].add_streaming_actor(event)
            elif (scope == "evaluate_ir_node" or etype == "Execute IR") and (
                ir_id := event.get("actor_ir_id") or event.get("ir_id")
            ) is not None:
                ir_id = str(ir_id) if isinstance(ir_id, int) else ir_id
                plan.stats[ir_id].add_execute_ir(event)
        return plan

    def render(self) -> str:
        if self.root_id is None:
            return self._render_flat()
        lines: list[str] = []
        self._render_node(self.root_id, "", lines)
        return "\n".join(lines)

    def _render_flat(self) -> str:
        if not self.stats:
            return "(no trace data found)"
        lines = ["(no query plan tree - showing flat stats by ir_id)"]
        by_type: dict[str, list[NodeStats]] = defaultdict(list)
        for stats in self.stats.values():
            by_type[stats.ir_type or "Unknown"].append(stats)
        for ir_type, items in sorted(by_type.items()):
            ann = []
            if rows := sum(s.rows or 0 for s in items):
                ann.append(f"rows={_fmt_count(rows)}")
            if chunks := sum(s.chunk_count for s in items):
                ann.append(f"chunks={chunks}")
            if nbytes := sum(s.total_bytes_output for s in items):
                ann.append(f"bytes_out={_fmt_bytes(nbytes)}")
            if ns := sum(s.total_duration_ns for s in items):
                ann.append(f"time={_fmt_duration(ns)}")
            if (workers := max((s.worker_count for s in items), default=0)) > 1:
                ann.append(f"workers={workers}")
            ann_str = f" [{', '.join(ann)}]" if ann else ""
            lines.append(f"  {ir_type.upper()}{ann_str}")
        return "\n".join(lines)

    def _render_node(self, ir_id: str | int, indent: str, lines: list[str]) -> None:
        node = self.nodes.get(ir_id, {})
        stats = self.stats.get(ir_id, NodeStats())
        ir_type = stats.ir_type or node.get("ir_type", "Unknown")
        props = node.get("properties", {})
        ann = []
        if stats.rows is not None:
            ann.append(f"rows={_fmt_count(stats.rows)}")
        if stats.chunk_count:
            ann.append(f"chunks={stats.chunk_count}")
        if stats.total_bytes_output:
            ann.append(f"bytes_out={_fmt_bytes(stats.total_bytes_output)}")
        if stats.total_duration_ns:
            ann.append(f"time={_fmt_duration(stats.total_duration_ns)}")
        if stats.decision:
            ann.append(f"decision={stats.decision}")
        if stats.worker_count > 1:
            ann.append(f"workers={stats.worker_count}")
        if stats.duplicated:
            ann.append("duplicated")
        if ir_type == "GroupBy" and (keys := props.get("keys")):
            ann.append(f"keys={tuple(keys)}")
        if ir_type == "Join" and (left_on := props.get("left_on")):
            ann.append(f"on={tuple(left_on)}")
        ann_str = f" [{', '.join(ann)}]" if ann else ""
        lines.append(f"{indent}{ir_type.upper()}{ann_str}")
        for child_id in node.get("children_ir_ids", []):
            self._render_node(child_id, indent + "  ", lines)


def _fmt_count(n: int) -> str:
    if n < 1_000:
        return str(n)
    if n < 1_000_000:
        return f"{n / 1_000:.2g}K"
    if n < 1_000_000_000:
        return f"{n / 1_000_000:.2g}M"
    return f"{n / 1_000_000_000:.2g}B"


def _fmt_bytes(n: int) -> str:
    if n < 1024:
        return f"{n}B"
    if n < 1024**2:
        return f"{n / 1024:.2g}KB"
    if n < 1024**3:
        return f"{n / 1024**2:.2g}MB"
    return f"{n / 1024**3:.2g}GB"


def _fmt_duration(ns: int) -> str:
    if ns < 1_000:
        return f"{ns}ns"
    if ns < 1_000_000:
        return f"{ns / 1_000:.2g}us"
    if ns < 1_000_000_000:
        return f"{ns / 1_000_000:.2g}ms"
    return f"{ns / 1_000_000_000:.2g}s"


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open() as f:
        return [json.loads(line) for line in f if line.strip()]


def _has_query_plan(traces: list[dict[str, Any]]) -> bool:
    return any(
        t.get("scope") == "plan"
        and (("plan" in t and t["plan"].get("nodes")) or t.get("nodes"))
        for t in traces
    )


def _iter_queries(
    records: list[dict[str, Any]],
) -> Iterator[tuple[int, int, dict[str, Any]]]:
    seen: set[tuple[int, int]] = set()
    for record in records:
        for qid_str, iters in record.get("records", {}).items():
            qid = int(qid_str)
            for i, it in enumerate(iters):
                if it.get("traces") and (qid, i) not in seen:
                    seen.add((qid, i))
                    yield qid, i, it


def get_traces_for_query(
    records: list[dict[str, Any]],
    query_id: int | None = None,
    iteration: int = 0,
) -> tuple[int, list[dict[str, Any]]] | None:
    # Scan all records (not deduplicated) so we can prefer a record that has
    # a query plan event, e.g. when multiple worker records cover the same query.
    fallback: tuple[int, list[dict[str, Any]]] | None = None
    for record in records:
        for qid_str, iters in record.get("records", {}).items():
            qid = int(qid_str)
            if (query_id is not None and qid != query_id) or iteration >= len(iters):
                continue
            traces = iters[iteration].get("traces")
            if not traces:
                continue
            if _has_query_plan(traces):
                return qid, traces
            fallback = fallback or (qid, traces)
    return fallback


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert benchmark traces to explain-like output."
    )
    parser.add_argument("input", type=Path)
    parser.add_argument("--query", "-q", type=int, default=None)
    parser.add_argument("--iteration", "-i", type=int, default=0)
    parser.add_argument("--list", "-l", action="store_true")
    parser.add_argument("--all", "-a", action="store_true")
    args = parser.parse_args()

    if not args.input.exists():
        print(f"Error: File not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    records = load_jsonl(args.input)

    if args.list:
        print("Available queries with traces:")
        for qid, i, it in _iter_queries(records):
            plan_flag = " [has plan]" if _has_query_plan(it["traces"]) else ""
            print(
                f"  Query {qid}, Iteration {i}: {it.get('duration', 0):.3f}s{plan_flag}"
            )
        return

    if args.all:
        for qid, i, it in _iter_queries(records):
            print(
                f"\n{'=' * 60}\nQuery {qid}, Iteration {i} (duration: {it.get('duration', 0):.3f}s)\n{'=' * 60}"
            )
            print(QueryPlan.from_traces(it["traces"]).render())
        return

    result = get_traces_for_query(records, args.query, args.iteration)
    if result is None:
        msg = (
            f"No traces found for query {args.query}, iteration {args.iteration}"
            if args.query is not None
            else "No traces found in the file"
        )
        print(f"Error: {msg}", file=sys.stderr)
        sys.exit(1)

    query_id, traces = result
    print(f"Query {query_id}, Iteration {args.iteration}\n{'=' * 40}")
    print(QueryPlan.from_traces(traces).render())
