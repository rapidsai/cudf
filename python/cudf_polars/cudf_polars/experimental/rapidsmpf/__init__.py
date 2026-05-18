# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""RapidsMPF streaming-engine support."""

from __future__ import annotations

# Side-effect imports: each module registers
# ``@generate_ir_sub_network.register(...)`` handlers at import time so the
# dispatch table is populated before any query is evaluated.
import cudf_polars.experimental.rapidsmpf.collectives.shuffle
import cudf_polars.experimental.rapidsmpf.collectives.sort
import cudf_polars.experimental.rapidsmpf.groupby
import cudf_polars.experimental.rapidsmpf.io
import cudf_polars.experimental.rapidsmpf.join
import cudf_polars.experimental.rapidsmpf.over
import cudf_polars.experimental.rapidsmpf.repartition
import cudf_polars.experimental.rapidsmpf.union  # noqa: F401

__all__: list[str] = []
