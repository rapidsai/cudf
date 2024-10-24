# Copyright (c) 2024, NVIDIA CORPORATION.

import pyarrow as pa
import pyarrow.compute as pc
import pytest
from utils import assert_column_eq

import pylibcudf as plc


@pytest.fixture(params=["float32", "float64"])
def column(request, has_nulls):
    values = [2.5, 2.49, 1.6, 8, -1.5, -1.7, -0.5, 0.5]
    typ = {"float32": pa.float32(), "float64": pa.float64()}[request.param]
    if has_nulls:
        values[2] = None
    return plc.interop.from_arrow(pa.array(values, type=typ))


@pytest.mark.parametrize(
    "round_mode", ["half_towards_infinity", "half_to_even"]
)
@pytest.mark.parametrize("decimals", [0, 1, 2, 5])
def test_round(column, round_mode, decimals):
    method = {
        "half_towards_infinity": plc.round.RoundingMethod.HALF_UP,
        "half_to_even": plc.round.RoundingMethod.HALF_EVEN,
    }[round_mode]
    got = plc.round.round(column, decimals, method)
    expect = pc.round(plc.interop.to_arrow(column), decimals, round_mode)

    assert_column_eq(expect, got)
