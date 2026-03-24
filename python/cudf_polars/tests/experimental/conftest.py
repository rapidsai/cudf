# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Configuration for StreamingEngine tests."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest
from rapidsmpf.bootstrap import get_nranks, is_running_with_rrun

from cudf_polars.experimental.rapidsmpf.frontend.spmd import spmd_execution

if TYPE_CHECKING:
    from collections.abc import Generator

    from cudf_polars.experimental.rapidsmpf.frontend.core import StreamingEngine


@pytest.fixture(autouse=True)
def _skip_unless_spmd(request: pytest.FixtureRequest) -> None:
    """Skip tests in SPMD multi-rank mode unless marked with ``pytest.mark.spmd``."""
    if (
        is_running_with_rrun()
        and get_nranks() > 1
        and not request.node.get_closest_marker("spmd")
    ):
        pytest.skip("skip: SPMD nranks > 1 (mark with pytest.mark.spmd to run)")


@pytest.fixture
def engine(
    request: pytest.FixtureRequest,
) -> Generator[StreamingEngine, None, None]:
    """Yield a :class:`~cudf_polars.experimental.rapidsmpf.frontend.spmd.SPMDEngine`.

    Default executor options enable dynamic planning with sensible partition
    sizes. Override options via ``indirect`` parametrization by passing a dict
    with ``"executor_options"`` and/or ``"engine_options"`` keys:

    .. code-block:: python

        @pytest.mark.parametrize(
            "engine",
            [{"executor_options": {"target_partition_size": 100_000_000}}],
            indirect=True,
        )
        def test_foo(engine): ...
    """
    params: dict[str, Any] = getattr(request, "param", {})
    executor_options = {
        "max_rows_per_partition": 50,
        "dynamic_planning": {},
        "target_partition_size": 1_000_000,
        **params.get("executor_options", {}),
    }
    with spmd_execution(
        executor_options=executor_options,
        engine_options=params.get("engine_options", {}),
    ) as engine:
        yield engine
