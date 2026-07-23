# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for Quent filesystem export."""

from __future__ import annotations

import json
import uuid
from typing import TYPE_CHECKING, Any

import pytest

from cudf_polars.quent._export import (
    EXTENSION,
    MODEL_QMI,
    SIDECAR_FILE_NAME,
    to_export_line,
    unwrap_event_data,
    write_quent_export,
)
from cudf_polars.quent._types import Engine, Network, Query, QueryGroup

if TYPE_CHECKING:
    from pathlib import Path


def _buffered_events() -> list[dict[str, Any]]:
    engine = Engine(id=uuid.UUID("019dd571-105a-7c53-a15b-713cbdd7666b"))
    query_group = QueryGroup(
        id=uuid.UUID("019dd571-1062-77c2-9803-62a66b6e0c5f"),
        instance_name="test-group",
    )
    query = Query(
        id=uuid.UUID("019dd571-1062-77c2-9803-62bd37658144"),
        instance_name="test-query",
    )
    network = Network(engine_id=engine.id)
    return [
        engine._init().to_dict(),
        query_group._declare(engine).to_dict(),
        query._init(query_group).to_dict(),
        network.declare().to_dict(),
        engine._exit().to_dict(),
    ]


def _read_ndjson_lines(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_unwrap_event_data() -> None:
    entity_name, payload = unwrap_event_data(
        {"Engine": {"Init": {"instance_name": "x"}}}
    )
    assert entity_name == "Engine"
    assert payload == {"Init": {"instance_name": "x"}}


def test_unwrap_event_data_rejects_multiple_wrappers() -> None:
    with pytest.raises(ValueError, match="exactly one entity wrapper"):
        unwrap_event_data({"Engine": {}, "Query": {}})


def test_unwrap_event_data_rejects_unknown_entity() -> None:
    with pytest.raises(ValueError, match="Unknown Quent entity type"):
        unwrap_event_data({"UnknownEntity": {}})


def test_to_export_line_unwraps_payload() -> None:
    event = {
        "id": "019dd571-105a-7c53-a15b-713cbdd7666b",
        "timestamp": 1777402450018164995,
        "data": {"Engine": {"Init": {"instance_name": "test"}}},
    }
    directory, export_line = to_export_line(event)
    assert directory == "engine"
    assert export_line == {
        "id": "019dd571-105a-7c53-a15b-713cbdd7666b",
        "timestamp": 1777402450018164995,
        "data": {"Init": {"instance_name": "test"}},
    }


def test_write_quent_export_creates_context_layout(tmp_path: Path) -> None:
    context_id = uuid.UUID("019dd571-105a-7c53-a15b-713cbdd7666b")
    events = _buffered_events()

    context_dir = write_quent_export(events, tmp_path, context_id)

    assert context_dir == tmp_path / str(context_id)
    assert (context_dir / SIDECAR_FILE_NAME).exists()
    assert (
        json.loads((context_dir / SIDECAR_FILE_NAME).read_text(encoding="utf-8"))
        == MODEL_QMI
    )

    expected_dirs = {"engine", "query_group", "query", "network"}
    created_dirs = {path.name for path in context_dir.iterdir() if path.is_dir()}
    assert created_dirs == expected_dirs

    for entity_dir in expected_dirs:
        stream_files = list((context_dir / entity_dir).glob(f"*.{EXTENSION}"))
        assert len(stream_files) == 1
        lines = _read_ndjson_lines(stream_files[0])
        assert lines
        for line in lines:
            assert "id" in line
            assert "timestamp" in line
            assert isinstance(line["data"], dict)
            assert len(line["data"]) == 1 or "seq" in line["data"]


def test_write_quent_export_unwraps_buffered_envelopes(tmp_path: Path) -> None:
    context_id = uuid.UUID("019dd571-105a-7c53-a15b-713cbdd7666b")
    events = _buffered_events()

    write_quent_export(events, tmp_path, context_id)

    engine_stream = next((tmp_path / str(context_id) / "engine").glob(f"*.{EXTENSION}"))
    engine_lines = _read_ndjson_lines(engine_stream)
    assert engine_lines[0]["data"] == {
        "Init": {
            "implementation": {
                "name": "cudf-polars",
                "version": engine_lines[0]["data"]["Init"]["implementation"]["version"],
                "custom_attributes": [],
            },
            "instance_name": "cudf-polars-019dd571",
        }
    }
    assert engine_lines[1]["data"] == {"Exit": None}

    network_stream = next(
        (tmp_path / str(context_id) / "network").glob(f"*.{EXTENSION}")
    )
    network_lines = _read_ndjson_lines(network_stream)
    assert network_lines[0]["data"] == {
        "Declaration": {
            "instance_name": "Network",
            "parent_group_id": str(context_id),
        }
    }


def test_write_quent_export_rejects_malformed_event(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="exactly one entity wrapper"):
        write_quent_export(
            [{"id": "x", "timestamp": 1, "data": {"Engine": {}, "Query": {}}}],
            tmp_path,
            uuid.uuid4(),
        )


def test_write_quent_traces_benchmark_writer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("structlog")
    from cudf_polars.streaming.benchmarks import utils as benchmark_utils

    run_id = uuid.UUID("019dd571-105a-7c53-a15b-713cbdd7666b")
    events = _buffered_events()

    class FakeEngine:
        _quent_events = events

    monkeypatch.chdir(tmp_path)
    benchmark_utils._write_quent_traces(
        FakeEngine(),  # type: ignore[arg-type]
        run_id,
        collect_traces=True,
    )

    context_dir = tmp_path / "logs" / str(run_id)
    assert (context_dir / SIDECAR_FILE_NAME).exists()
    assert (context_dir / "engine").is_dir()
    assert (context_dir / "query").is_dir()
