# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Miscellaneous conversion functions."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from cudf_polars.typing import Slice


def from_polars_slice(zlice: Slice, *, num_rows: int) -> list[int]:
    """
    Convert a Polar slice into something pylibcudf handles.

    Parameters
    ----------
    zlice
         The slice to convert
    num_rows
         The number of rows in the object being sliced.

    Returns
    -------
    List of start and end slice bounds.
    """
    start, length = zlice
    if length is None:
        length = num_rows
    if start < 0:
        start += num_rows
    # Polars implementation wraps negative start by num_rows, then
    # adds length to start to get the end, then clamps both to
    # [0, num_rows)
    end = start + length
    start = max(min(start, num_rows), 0)
    end = max(min(end, num_rows), 0)
    return [start, end]
