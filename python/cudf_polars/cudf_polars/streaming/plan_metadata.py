# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Metadata produced while optimizing a query plan."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from cudf_polars.dsl.ir import IR


@dataclass
class PlanMetadata:
    """Typed details associated with nodes in an optimized IR graph."""

    _operator_details: dict[IR, list[object]] = field(
        default_factory=lambda: defaultdict(list)
    )

    def add_operator_detail(self, node: IR, detail: object) -> None:
        """Associate one detail payload with an operator."""
        self._operator_details[node].append(detail)

    def operator_details(self, node: IR) -> tuple[object, ...]:
        """Return detail payloads associated with an operator."""
        return tuple(self._operator_details.get(node, ()))
