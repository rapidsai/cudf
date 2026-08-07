# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
"""cudfgrep: a GPU-accelerated grep utility built on cuDF."""

from __future__ import annotations

from cudf.grep._grep import grep, main

__all__ = ["grep", "main"]
