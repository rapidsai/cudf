# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Common utilities for collective operations."""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING, Literal

from rapidsmpf.shuffler import Shuffler

from cudf_polars.dsl.ir import Distinct, GroupBy
from cudf_polars.dsl.traversal import traversal
from cudf_polars.experimental.join import Join
from cudf_polars.experimental.repartition import Repartition
from cudf_polars.experimental.shuffle import Shuffle

if TYPE_CHECKING:
    from types import TracebackType

    from cudf_polars.dsl.ir import IR
    from cudf_polars.utils.config import ConfigOptions


# Set of available collective IDs
_collective_id_vacancy: set[int] = set(range(Shuffler.max_concurrent_shuffles))
_collective_id_vacancy_lock: threading.Lock = threading.Lock()


def _get_new_collective_id() -> int:
    with _collective_id_vacancy_lock:
        if not _collective_id_vacancy:
            raise ValueError(
                f"Cannot shuffle more than {Shuffler.max_concurrent_shuffles} "
                "times in a single query."
            )

        return _collective_id_vacancy.pop()


def _release_collective_id(collective_id: int) -> None:
    """Release a collective ID back to the vacancy set."""
    with _collective_id_vacancy_lock:
        _collective_id_vacancy.add(collective_id)


class ReserveOpIDs:
    """
    Context manager to reserve collective IDs for pipeline execution.

    Parameters
    ----------
    ir : IR
        The root IR node of the pipeline.
    config_options : ConfigOptions, optional
        Configuration options (needed for dynamic planning).

    Notes
    -----
    This context manager:
    1. Identifies all IR nodes that may require collective operations
    2. Reserves collective IDs from the vacancy pool
    3. Creates a mapping from IR nodes to their reserved IDs
    4. Releases all IDs back to the pool on __exit__

    Each IR node may require multiple collective operation IDs
    (e.g., for metadata gathering, shuffling multiple sides of a join).
    """

    def __init__(self, ir: IR, config_options: ConfigOptions | None = None):
        self.config_options = config_options

        # Check if dynamic planning is enabled
        self.dynamic_planning_enabled = (
            config_options is not None
            and config_options.executor.name == "streaming"
            and config_options.executor.dynamic_planning is not None
        )

        # Find all collective IR nodes.
        collective_types: tuple[type, ...] = (Shuffle, Join, Repartition)
        if self.dynamic_planning_enabled:
            # Include GroupBy and Distinct when dynamic planning is enabled
            collective_types = (Shuffle, Join, Repartition, GroupBy, Distinct)

        self.collective_nodes: list[IR] = [
            node for node in traversal([ir]) if isinstance(node, collective_types)
        ]
        self.collective_id_map: dict[IR, list[int]] = {}

    def __enter__(self) -> dict[IR, list[int]]:
        """
        Reserve collective IDs and return the mapping.

        Returns
        -------
        collective_id_map : dict[IR, list[int]]
            Mapping from IR nodes to their reserved collective IDs.
            Each IR node gets a list of IDs to support multiple
            collective operations per node.
        """
        # Reserve IDs and map nodes to a list of IDs
        for node in self.collective_nodes:
            if isinstance(node, (GroupBy, Distinct)) and self.dynamic_planning_enabled:
                # GroupBy/Distinct need 2 IDs: one for size allgather, one for shuffle
                self.collective_id_map[node] = [
                    _get_new_collective_id(),
                    _get_new_collective_id(),
                ]
            elif isinstance(node, Join) and self.dynamic_planning_enabled:
                # Join needs 3 IDs: size allgather, left shuffle/bcast, right shuffle/bcast
                self.collective_id_map[node] = [
                    _get_new_collective_id(),
                    _get_new_collective_id(),
                    _get_new_collective_id(),
                ]
            else:
                self.collective_id_map[node] = [_get_new_collective_id()]

        return self.collective_id_map

    def __exit__(
        self,
        exc_type: type | None,
        exc_val: Exception | None,
        exc_tb: TracebackType | None,
    ) -> Literal[False]:
        """Release all reserved collective IDs back to the vacancy pool."""
        for collective_ids in self.collective_id_map.values():
            for collective_id in collective_ids:
                _release_collective_id(collective_id)
        return False
