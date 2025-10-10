# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Testing for the RAPIDS-MPF streaming engine."""

from __future__ import annotations

import pytest

# Skip if rapidsmpf is not installed
pytest.importorskip("rapidsmpf")

__all__: list[str] = []
