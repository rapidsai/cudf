# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from cudf_polars.utils.cuda_stream import (
    get_stream_for_conditional_join_predicate,
)


def test_get_stream_for_conditional_join_predicate():
    stream = get_stream_for_conditional_join_predicate()
    assert stream is get_stream_for_conditional_join_predicate()
