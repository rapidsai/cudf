# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

import pyarrow as pa
import pyarrow.compute as pc
import pytest
from utils import assert_column_eq

import pylibcudf as plc


@pytest.fixture(params=["int32", "int64"])
def column(request, has_nulls):
    values = [250, 249, 160, 800, -150, -170, -50, 50]
    typ = {"int32": pa.int32(), "int64": pa.int64()}[request.param]
    if has_nulls:
        values[2] = None
    return plc.Column.from_arrow(pa.array(values, type=typ))


@pytest.mark.parametrize(
    "round_mode", ["half_towards_infinity", "half_to_even"]
)
@pytest.mark.parametrize("decimals", [0, -1, -2])
def test_round(column, round_mode, decimals):
    method = {
        "half_towards_infinity": plc.round.RoundingMethod.HALF_UP,
        "half_to_even": plc.round.RoundingMethod.HALF_EVEN,
    }[round_mode]
    got = plc.round.round(column, decimals, method)
    expect = pc.round(column.to_arrow(), decimals, round_mode)

    assert_column_eq(expect, got)
