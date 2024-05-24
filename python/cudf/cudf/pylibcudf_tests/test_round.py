# Copyright (c) 2024, NVIDIA CORPORATION.

import pyarrow as pa
import pytest
from utils import assert_column_eq

import cudf._lib.pylibcudf as plc


@pytest.fixture(scope="module")
def pa_round_data():
    pa_arr = pa.array([1.5, 2.5, 1.35, 1.45, 15, 25], type=pa.float64())
    return pa_arr


@pytest.fixture(scope="module")
def plc_round_data(pa_round_data):
    return plc.interop.from_arrow(pa_round_data)


@pytest.mark.parametrize("decimal_places", [0, 1, 10])
@pytest.mark.parametrize(
    "round_mode",
    [
        ("half_up", plc.round.RoundingMethod.HALF_UP),
        ("half_to_even", plc.round.RoundingMethod.HALF_EVEN),
    ],
)
def test_round(pa_round_data, plc_round_data, decimal_places, round_mode):
    pa_round_mode, plc_round_mode = round_mode
    res = plc.round.round(
        plc_round_data,
        decimal_places=decimal_places,
        round_method=plc_round_mode,
    )
    expected = pa.compute.round(
        pa_round_data, ndigits=decimal_places, round_mode=pa_round_mode
    )

    assert_column_eq(res, expected)
