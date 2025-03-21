# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import pytest

import polars as pl
from polars.testing import assert_frame_equal

from cudf_polars.utils.versions import POLARS_VERSION_LT_125


@pytest.mark.skipif(
    POLARS_VERSION_LT_125, reason="Profiling with GPU engine requires polars>=1.25"
)
def test_profile_basic():
    df = pl.LazyFrame(
        {
            "a": [1, 2, 1, 3, 5, None, None],
            "b": [1.5, 2.5, None, 1.5, 3, float("nan"), 3],
            "c": [True, True, True, True, False, False, True],
            "d": [1, 2, -1, 10, 6, -1, -7],
        }
    )

    q = df.sort("a").group_by("a", pl.col("b")).agg(pl.col("d").sum())

    result, timings = q.profile(engine="gpu")

    assert "gpu-ir-translation" in timings["node"]

    assert_frame_equal(result, q.collect(engine="in-memory"), check_row_order=False)
