# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Dask sizeof."""

from __future__ import annotations

from dask.sizeof import sizeof as sizeof_dispatch

from cudf_polars.containers import Column, DataFrame


@sizeof_dispatch.register(Column)
def _(x: Column) -> int:
    ret = 0
    if x.obj.data() is not None:
        ret += x.obj.data().nbytes
    if x.obj.null_mask() is not None:
        ret += x.obj.null_mask().nbytes
    if x.obj.children() is not None:
        ret += sum(sizeof_dispatch(c) for c in x.obj.children())
    return ret


@sizeof_dispatch.register(DataFrame)
def _(x: DataFrame) -> int:
    return sum(sizeof_dispatch(c) for c in x.columns)
