# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Memory reservations for the RapidsMPF streaming runtime."""

from __future__ import annotations

import enum
from typing import TYPE_CHECKING

from rapidsmpf.memory.buffer import MemoryType
from rapidsmpf.streaming.core.memory_reserve_or_wait import reserve_memory

if TYPE_CHECKING:
    import cudf_polars_quent as quent_bindings

    from rapidsmpf.memory.memory_reservation import MemoryReservation
    from rapidsmpf.streaming.core.context import Context

    from cudf_polars.dsl.ir import IRExecutionContext

__all__ = ["MemoryReservationPurpose", "reserve_memory_traced"]


class MemoryReservationPurpose(enum.StrEnum):
    """Why an actor requested memory admission."""

    PYTHON_SCAN = "python-scan"
    SCAN = "scan"
    BROADCAST_JOIN = "broadcast-join"
    JOIN = "join"
    PREFILTER_PROJECT_KEYS = "prefilter-project-keys"
    ALLGATHER_EXTRACT = "allgather-extract"
    ORDERING_UNPACK_REMOTE = "ordering-unpack-remote"
    SHUFFLE_INSERT_HASH = "shuffle-insert-hash"
    SHUFFLE_INSERT_HASH_KEYS = "shuffle-insert-hash-keys"
    SHUFFLE_INSERT_SPLIT = "shuffle-insert-split"
    SHUFFLE_INSERT_INDEX = "shuffle-insert-index"
    SHUFFLE_EXTRACT = "shuffle-extract"
    REPARTITION_EXTRACT = "repartition-extract"


async def reserve_memory_traced(
    context: Context,
    size: int,
    *,
    net_memory_delta: int,
    ir_context: IRExecutionContext | None,
    purpose: MemoryReservationPurpose,
    sequence_number: int | None = None,
    mem_type: MemoryType = MemoryType.DEVICE,
    allow_overbooking: bool | None = None,
) -> MemoryReservation:
    """Reserve memory and trace the time spent waiting for admission."""
    quent_state = None if ir_context is None else ir_context.quent_ir_execution_state
    handle: quent_bindings.MemoryReservationRequestedHandle | None = None
    if quent_state is not None:
        import cudf_polars_quent as _quent

        assert quent_state.actor_id is not None, (
            "Memory reservations must be emitted from an Actor scope"
        )
        reservation_id = _quent.now_v7()
        handle = (
            quent_state.query_worker_state.runtime.session.binding_context.memory_reservation_observer()
            .handle(reservation_id)
            .requested(
                instance_name=(
                    f"reserve-{purpose.value}-{quent_state.operator_id.hex[:8]}-"
                    f"{reservation_id.hex[:8]}"
                ),
                actor=quent_state.actor_id,
                request={
                    "purpose": purpose.value,
                    "size_bytes": size,
                    "memory_type": mem_type.name,
                    "net_memory_delta": net_memory_delta,
                    "allow_overbooking": allow_overbooking,
                    "sequence_number": sequence_number,
                },
            )
        )

    try:
        reservation = await reserve_memory(
            context,
            size=size,
            net_memory_delta=net_memory_delta,
            mem_type=mem_type,
            allow_overbooking=allow_overbooking,
        )
    except BaseException as error:
        if handle is not None:
            handle.failed(error=str(error))
        raise
    else:
        if handle is not None:
            handle.granted()
        return reservation
