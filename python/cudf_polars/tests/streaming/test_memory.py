# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for traced streaming-runtime memory reservations."""

from __future__ import annotations

import uuid
from unittest.mock import AsyncMock, MagicMock

import pytest

from cudf_polars.streaming.actor_graph import memory


@pytest.fixture
def traced_reservation(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[MagicMock, MagicMock, AsyncMock, MagicMock]:
    handle = MagicMock()
    observer = MagicMock()
    requested = observer.handle.return_value.requested
    requested.return_value = handle
    state = MagicMock(
        actor_id=uuid.uuid4(),
        operator_id=uuid.uuid4(),
    )
    binding_context = state.query_worker_state.runtime.session.binding_context
    binding_context.memory_reservation_observer.return_value = observer
    ir_context = MagicMock(quent_ir_execution_state=state)
    reserve = AsyncMock()
    monkeypatch.setattr(memory, "reserve_memory", reserve)
    return ir_context, handle, reserve, requested


@pytest.mark.asyncio
async def test_traced_reservation_granted(
    traced_reservation: tuple[MagicMock, MagicMock, AsyncMock, MagicMock],
) -> None:
    ir_context, handle, reserve, requested = traced_reservation
    reservation = MagicMock()
    reserve.return_value = reservation
    handle.granted.side_effect = RuntimeError("telemetry failed")

    with pytest.raises(RuntimeError, match="telemetry failed"):
        await memory.reserve_memory_traced(
            MagicMock(),
            20,
            net_memory_delta=10,
            ir_context=ir_context,
            purpose=memory.MemoryReservationPurpose.SCAN,
            sequence_number=2,
            allow_overbooking=False,
        )

    assert requested.call_args.kwargs["request"] == {
        "purpose": "scan",
        "size_bytes": 20,
        "memory_type": "DEVICE",
        "net_memory_delta": 10,
        "allow_overbooking": False,
        "sequence_number": 2,
    }
    handle.granted.assert_called_once_with()
    handle.failed.assert_not_called()


@pytest.mark.asyncio
async def test_traced_reservation_start_failure_is_raised(
    traced_reservation: tuple[MagicMock, MagicMock, AsyncMock, MagicMock],
) -> None:
    ir_context, handle, reserve, requested = traced_reservation
    reserve.return_value = MagicMock()
    requested.side_effect = RuntimeError("telemetry failed")

    with pytest.raises(RuntimeError, match="telemetry failed"):
        await memory.reserve_memory_traced(
            MagicMock(),
            20,
            net_memory_delta=10,
            ir_context=ir_context,
            purpose=memory.MemoryReservationPurpose.SCAN,
        )

    reserve.assert_not_awaited()
    handle.granted.assert_not_called()
    handle.failed.assert_not_called()


@pytest.mark.asyncio
async def test_traced_reservation_failed(
    traced_reservation: tuple[MagicMock, MagicMock, AsyncMock, MagicMock],
) -> None:
    ir_context, handle, reserve, _ = traced_reservation
    reserve.side_effect = RuntimeError("admission failed")
    handle.failed.side_effect = RuntimeError("telemetry failed")

    with pytest.raises(RuntimeError, match="telemetry failed"):
        await memory.reserve_memory_traced(
            MagicMock(),
            20,
            net_memory_delta=10,
            ir_context=ir_context,
            purpose=memory.MemoryReservationPurpose.SCAN,
        )

    handle.failed.assert_called_once_with(error="admission failed")
    handle.granted.assert_not_called()
