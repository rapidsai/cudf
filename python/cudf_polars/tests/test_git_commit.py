# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import cudf_polars


def test_git_commit():
    assert cudf_polars.__git_commit__ is not None
    assert cudf_polars.__git_commit__ != ""
