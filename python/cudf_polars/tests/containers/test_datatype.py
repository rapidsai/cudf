# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import polars as pl

from cudf_polars.containers import DataType


def test_hash():
    dtype = pl.Int8()
    assert hash(dtype) == hash(DataType(dtype))


def test_eq():
    dtype = pl.Int8()
    data_type = DataType(dtype)

    assert data_type != dtype
    assert data_type == DataType(dtype)


def test_repr():
    data_type = DataType(pl.Int8())

    assert repr(data_type) == "<DataType(polars=Int8, plc=<type_id.INT8: 1>)>"
