# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import pytest

from cudf_polars.experimental.benchmarks.explain_trace import (
    QueryPlan,
    get_traces_for_query,
)
from cudf_polars.experimental.benchmarks.explain_trace._core import (
    NodeStats,
    _fmt_bytes,
    _fmt_count,
    _fmt_duration,
    _has_query_plan,
)

_PLAN_NEW = {
    "scope": "plan",
    "plan": {
        "nodes": {
            "root": {
                "type": "Join",
                "children": ["left", "right"],
                "properties": {"left_on": ["key"]},
            },
            "left": {"type": "HStack", "children": [], "properties": {}},
            "right": {"type": "Filter", "children": [], "properties": {}},
        },
        "roots": ["root"],
    },
}

_PLAN_OLD = {
    "scope": "plan",
    "nodes": [
        {
            "ir_id": 1,
            "ir_type": "Join",
            "children_ir_ids": [2, 3],
            "properties": {"left_on": ["key"]},
        },
        {"ir_id": 2, "ir_type": "HStack", "children_ir_ids": [], "properties": {}},
        {"ir_id": 3, "ir_type": "Filter", "children_ir_ids": [], "properties": {}},
    ],
}


def _actor(ir_id, ir_type, chunk_count, rows, *, duplicated=False, decision=None):
    e = {
        "scope": "actor",
        "actor_ir_id": ir_id,
        "actor_ir_type": ir_type,
        "chunk_count": chunk_count,
        "rows": rows,
        "duplicated": duplicated,
    }
    if decision:
        e["decision"] = decision
    return e


def _execute_ir(
    ir_id,
    ir_type,
    total_bytes_input,
    total_bytes_output,
    start=1_000_000,
    stop=2_000_000,
):
    return {
        "scope": "evaluate_ir_node",
        "event": "Execute IR",
        "actor_ir_id": ir_id,
        "type": ir_type,
        "total_bytes_input": total_bytes_input,
        "total_bytes_output": total_bytes_output,
        "start": start,
        "stop": stop,
    }


def test_add_streaming_actor_partitioned():
    s = NodeStats()
    s.add_streaming_actor(_actor("a", "HStack", chunk_count=3, rows=100))
    s.add_streaming_actor(_actor("a", "HStack", chunk_count=2, rows=50))
    assert s.chunk_count == 5
    assert s.rows == 150
    assert s.worker_count == 2
    assert not s.duplicated


def test_add_streaming_actor_duplicated():
    s = NodeStats()
    s.add_streaming_actor(
        _actor("a", "HStack", chunk_count=3, rows=100, duplicated=True)
    )
    s.add_streaming_actor(
        _actor("a", "HStack", chunk_count=3, rows=100, duplicated=True)
    )
    assert s.chunk_count == 3
    assert s.rows == 100
    assert s.duplicated


def test_add_streaming_actor_decision():
    s = NodeStats()
    s.add_streaming_actor(
        _actor("a", "Join", chunk_count=1, rows=10, decision="broadcast_left")
    )
    assert s.decision == "broadcast_left"


def test_add_execute_ir():
    s = NodeStats()
    s.add_execute_ir(
        _execute_ir(
            "a",
            "Filter",
            total_bytes_input=1000,
            total_bytes_output=500,
            start=1_000_000,
            stop=3_000_000,
        )
    )
    s.add_execute_ir(
        _execute_ir(
            "a",
            "Filter",
            total_bytes_input=800,
            total_bytes_output=400,
            start=1_000_000,
            stop=2_000_000,
        )
    )
    assert s.total_bytes_input == 1800
    assert s.total_bytes_output == 900
    assert s.total_duration_ns == 3_000_000
    assert s.exec_count == 2


def test_from_traces_new_format():
    plan = QueryPlan.from_traces([_PLAN_NEW])
    assert plan.root_id == "root"
    assert plan.nodes["root"]["ir_type"] == "Join"
    assert plan.nodes["root"]["children_ir_ids"] == ["left", "right"]
    assert plan.stats["root"].ir_type == "Join"


def test_from_traces_old_format():
    plan = QueryPlan.from_traces([_PLAN_OLD])
    assert plan.root_id == 1
    assert plan.nodes[1]["ir_type"] == "Join"
    assert plan.nodes[1]["children_ir_ids"] == [2, 3]


def test_from_traces_actor_int_ir_id():
    # Integer ir_ids from actor events should be coerced to str
    plan = QueryPlan.from_traces(
        [
            _PLAN_NEW,
            _actor("root", "Join", chunk_count=4, rows=200, decision="shuffle"),
        ]
    )
    assert plan.stats["root"].rows == 200
    assert plan.stats["root"].decision == "shuffle"


def test_from_traces_execute_ir():
    plan = QueryPlan.from_traces(
        [
            _PLAN_NEW,
            _execute_ir(
                "left", "HStack", total_bytes_input=512, total_bytes_output=256
            ),
        ]
    )
    assert plan.stats["left"].total_bytes_output == 256
    assert plan.stats["left"].exec_count == 1


def test_render_tree():
    traces = [
        _PLAN_NEW,
        _actor("root", "Join", chunk_count=4, rows=200, decision="shuffle"),
        _actor("left", "HStack", chunk_count=2, rows=100),
        _actor("right", "Filter", chunk_count=2, rows=100),
    ]
    output = QueryPlan.from_traces(traces).render()
    lines = output.splitlines()
    assert lines[0].startswith("JOIN")
    assert "decision=shuffle" in lines[0]
    assert "on=('key',)" in lines[0]
    assert lines[1].startswith("  HSTACK")
    assert lines[2].startswith("  FILTER")


def test_render_flat_fallback():
    traces = [_actor("x", "GroupBy", chunk_count=3, rows=50)]
    output = QueryPlan.from_traces(traces).render()
    assert "no query plan tree" in output
    assert "GROUPBY" in output


def test_render_empty():
    assert QueryPlan().render() == "(no trace data found)"


def test_has_query_plan_new_format():
    assert _has_query_plan([_PLAN_NEW])


def test_has_query_plan_old_format():
    assert _has_query_plan([_PLAN_OLD])


def test_has_query_plan_false():
    assert not _has_query_plan([_actor("a", "HStack", 1, 10)])


def _make_records(queries: dict) -> list[dict]:
    return [{"records": {str(qid): iters for qid, iters in queries.items()}}]


def test_get_traces_prefers_plan():
    # Two worker records for the same (qid=4, iteration=0): only one has a plan.
    no_plan = [_actor("a", "HStack", 1, 10)]
    with_plan = [_PLAN_NEW, _actor("root", "Join", 2, 20)]
    records = [
        {"records": {"4": [{"traces": no_plan}]}},
        {"records": {"4": [{"traces": with_plan}]}},
    ]
    qid, traces = get_traces_for_query(records, query_id=4, iteration=0)
    assert qid == 4
    assert _has_query_plan(traces)


def test_get_traces_fallback_no_plan():
    traces = [_actor("a", "HStack", 1, 10)]
    records = _make_records({4: [{"traces": traces}]})
    result = get_traces_for_query(records, query_id=4, iteration=0)
    assert result is not None
    assert result[0] == 4


def test_get_traces_not_found():
    records = _make_records({4: [{"traces": [_actor("a", "HStack", 1, 10)]}]})
    assert get_traces_for_query(records, query_id=9) is None


def test_get_traces_first_query():
    records = _make_records({4: [{"traces": [_PLAN_NEW]}]})
    qid, _ = get_traces_for_query(records)
    assert qid == 4


@pytest.mark.parametrize(
    "n,expected",
    [
        (0, "0"),
        (999, "999"),
        (1500, "1.5K"),
        (1_500_000, "1.5M"),
        (2_000_000_000, "2B"),
    ],
)
def test_fmt_count(n, expected):
    assert _fmt_count(n) == expected


@pytest.mark.parametrize(
    "n,expected",
    [
        (512, "512B"),
        (2048, "2KB"),
        (2 * 1024**2, "2MB"),
        (3 * 1024**3, "3GB"),
    ],
)
def test_fmt_bytes(n, expected):
    assert _fmt_bytes(n) == expected


@pytest.mark.parametrize(
    "ns,expected",
    [
        (500, "500ns"),
        (5_000, "5us"),
        (5_000_000, "5ms"),
        (5_000_000_000, "5s"),
    ],
)
def test_fmt_duration(ns, expected):
    assert _fmt_duration(ns) == expected
