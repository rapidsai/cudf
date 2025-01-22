# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Multi-partition base classes."""

from __future__ import annotations

from typing import TYPE_CHECKING

from cudf_polars.dsl.ir import Union

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence

    from cudf_polars.containers import DataFrame
    from cudf_polars.dsl.expr import NamedExpr
    from cudf_polars.dsl.nodebase import Node


class PartitionInfo:
    """Partitioning information."""

    __slots__ = ("count", "partitioned_on")
    count: int
    """Partition count."""
    partitioned_on: tuple[NamedExpr, ...]
    """Columns the data is hash-partitioned on."""

    def __init__(
        self,
        count: int,
        partitioned_on: tuple[NamedExpr, ...] = (),
    ):
        self.count = count
        self.partitioned_on = partitioned_on

    def keys(self, node: Node) -> Iterator[tuple[str, int]]:
        """Return the partitioned keys for a given node."""
        name = get_key_name(node)
        yield from ((name, i) for i in range(self.count))


def get_key_name(node: Node) -> str:
    """Generate the key name for a Node."""
    return f"{type(node).__name__.lower()}-{hash(node)}"


def _concat(dfs: Sequence[DataFrame]) -> DataFrame:
    # Concatenate a sequence of DataFrames vertically
    return Union.do_evaluate(None, *dfs)
