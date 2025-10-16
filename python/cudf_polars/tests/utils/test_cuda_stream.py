# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from cudf_polars.utils.cuda_stream import (
    get_stream_for_conditional_join_predicate,
    get_stream_for_offset_windows,
    get_stream_for_stats,
)


def test_get_stream_for_offset_windows():
    stream = get_stream_for_offset_windows()
    assert stream is get_stream_for_offset_windows()


def test_get_stream_for_stats():
    stream = get_stream_for_stats()
    assert stream is get_stream_for_stats()


def test_get_stream_for_conditional_join_predicate():
    stream = get_stream_for_conditional_join_predicate()
    assert stream is get_stream_for_conditional_join_predicate()
