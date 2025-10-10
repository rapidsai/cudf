# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Union logic for the RapidsMPF streaming engine."""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING, Any

from rapidsmpf.shuffler import Shuffler
from rapidsmpf.streaming.core.channel import Channel
from rapidsmpf.streaming.cudf.partition import partition_and_pack, unpack_and_concat
from rapidsmpf.streaming.cudf.shuffler import shuffler

from cudf_polars.dsl.expr import Col
from cudf_polars.experimental.rapidsmpf.dispatch import (
    generate_ir_sub_network,
)
from cudf_polars.experimental.shuffle import Shuffle

if TYPE_CHECKING:
    from cudf_polars.dsl.ir import IR
    from cudf_polars.experimental.rapidsmpf.core import SubNetGenerator


# Set of available shuffle IDs
_shuffle_id_vacancy: set[int] = set(range(Shuffler.max_concurrent_shuffles))
_shuffle_id_vacancy_lock: threading.Lock = threading.Lock()


def _get_new_shuffle_id() -> int:
    with _shuffle_id_vacancy_lock:
        if not _shuffle_id_vacancy:
            raise ValueError(
                f"Cannot shuffle more than {Shuffler.max_concurrent_shuffles} "
                "times in a single query."
            )

        return _shuffle_id_vacancy.pop()


@generate_ir_sub_network.register(Shuffle)
def _(
    ir: Shuffle, rec: SubNetGenerator
) -> tuple[dict[IR, list[Any]], dict[IR, list[Any]]]:
    # Local shuffle operation.
    # TODO: How to distinguish between local and global shuffle?
    # May need to track two different contexts?

    # Process children
    (child,) = ir.children
    nodes, channels = rec(child)

    keys: list[Col] = [ne.value for ne in ir.keys if isinstance(ne.value, Col)]
    if len(keys) != len(ir.keys):  # pragma: no cover
        raise NotImplementedError("Shuffle requires simple keys.")
    column_names = list(ir.schema.keys())

    context = rec.state["ctx"]
    columns_to_hash = tuple(column_names.index(k.name) for k in keys)
    num_partitions = rec.state["partition_info"][ir].count
    op_id = _get_new_shuffle_id()

    # Partition and pack
    ch1 = channels[child].pop()
    ch2 = Channel()
    nodes[ir] = []
    nodes[ir].append(
        partition_and_pack(
            context,
            ch_in=ch1,
            ch_out=ch2,
            columns_to_hash=columns_to_hash,
            num_partitions=num_partitions,
        )
    )

    # Shuffle
    ch3 = Channel()
    nodes[ir].append(
        shuffler(
            context,
            ch_in=ch2,
            ch_out=ch3,
            op_id=op_id,
            total_num_partitions=num_partitions,
        )
    )

    # Unpack and concat
    ch4 = Channel()
    nodes[ir].append(unpack_and_concat(context, ch_in=ch3, ch_out=ch4))
    channels[ir] = [ch4]

    return nodes, channels
