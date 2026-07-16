# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Export Quent events to the filesystem directory layout."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from cudf_polars.quent._types import EventName, new_quent_id

if TYPE_CHECKING:
    import uuid
    from pathlib import Path


SIDECAR_FILE_NAME = "model.qmi"
EXTENSION = "ndjson"

MODEL_QMI: dict[str, Any] = {
    "quent": {
        "version": "0.1.0",
        "commit": "9c2924510756d9eca288819e125330805053051f",
        "remote": "https://github.com/rapidsai/quent",
    },
    "model": {
        "name": "Simulator",
        "package": "quent-simulator-instrumentation",
        "type_path": "quent_simulator_instrumentation::SimulatorEvent",
        "source": {
            "version": "0.1.0",
            "commit": "9c2924510756d9eca288819e125330805053051f",
            "remote": "https://github.com/rapidsai/quent",
        },
        "analyzer_package": "quent-simulator-analyzer",
    },
}

ENTITY_DIRECTORIES: dict[str, str] = {
    EventName.ENGINE.value: "engine",
    EventName.WORKER.value: "worker",
    EventName.QUERY_GROUP.value: "query_group",
    EventName.QUERY.value: "query",
    EventName.PLAN.value: "plan",
    EventName.OPERATOR.value: "operator",
    EventName.PORT.value: "port",
    EventName.TASK.value: "task",
    EventName.MEMORY.value: "memory",
    EventName.CHANNEL.value: "channel",
    EventName.THREAD_POOL.value: "thread_pool",
    EventName.PROCESSOR.value: "processor",
    EventName.NETWORK.value: "network",
}


def unwrap_event_data(data: dict[str, Any]) -> tuple[str, Any]:
    """
    Extract the entity name and unwrapped payload from a buffered event.

    Buffered events wrap payloads as ``{"Engine": {...}}``; directory export
    stores the payload directly because the entity type is implied by the
    subdirectory name.
    """
    if len(data) != 1:
        msg = (
            "Expected event data with exactly one entity wrapper, "
            f"got {len(data)} keys: {sorted(data)}"
        )
        raise ValueError(msg)
    entity_name, payload = next(iter(data.items()))
    if entity_name not in ENTITY_DIRECTORIES:
        msg = f"Unknown Quent entity type: {entity_name!r}"
        raise ValueError(msg)
    return entity_name, payload


def to_export_line(event: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    """Convert a buffered event envelope to directory export line format."""
    entity_name, payload = unwrap_event_data(event["data"])
    directory = ENTITY_DIRECTORIES[entity_name]
    export_line = {
        "id": event["id"],
        "timestamp": event["timestamp"],
        "data": payload,
    }
    return directory, export_line


def write_sidecar(context_dir: Path, sidecar: dict[str, Any]) -> None:
    """Atomically write the ``model.qmi`` provenance sidecar."""
    tmp_path = context_dir / f".{SIDECAR_FILE_NAME}.tmp"
    final_path = context_dir / SIDECAR_FILE_NAME
    tmp_path.write_text(json.dumps(sidecar, indent=2) + "\n", encoding="utf-8")
    tmp_path.replace(final_path)


def write_quent_export(
    events: list[dict[str, Any]],
    export_root: Path,
    context_id: uuid.UUID,
    *,
    sidecar: dict[str, Any] | None = None,
) -> Path:
    """
    Write Quent events to the filesystem export layout.

    Parameters
    ----------
    events
        Buffered Quent event envelopes from ``engine._quent_events``.
    export_root
        Root directory for exported contexts (e.g. ``logs``).
    context_id
        Context UUID, typically the engine/run id.
    sidecar
        Optional provenance payload for ``model.qmi``. Defaults to
        :data:`MODEL_QMI`.

    Returns
    -------
    Path
        The context directory ``export_root/<context_id>/``.
    """
    context_dir = export_root / str(context_id)
    context_dir.mkdir(parents=True, exist_ok=True)

    write_sidecar(context_dir, sidecar or MODEL_QMI)

    grouped: dict[str, list[dict[str, Any]]] = {}
    for event in events:
        directory, export_line = to_export_line(event)
        grouped.setdefault(directory, []).append(export_line)

    for directory, lines in grouped.items():
        entity_dir = context_dir / directory
        entity_dir.mkdir(parents=True, exist_ok=True)
        stream_path = entity_dir / f"{new_quent_id()}.{EXTENSION}"
        with stream_path.open("w", encoding="utf-8") as stream_file:
            for line in lines:
                stream_file.write(json.dumps(line, separators=(",", ":")))
                stream_file.write("\n")

    return context_dir
