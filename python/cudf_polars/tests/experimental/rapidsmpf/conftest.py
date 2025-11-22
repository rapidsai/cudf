# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Configuration for rapidsmpf tests."""

from __future__ import annotations

import pytest

# Skip all tests in this directory if rapidsmpf is not available
pytest.importorskip("rapidsmpf")
