# Copyright (c) 2024, NVIDIA CORPORATION.

import pyarrow as pa
import pylibcudf as plc
import pytest


@pytest.fixture(scope="module")
def input_data():
    input_arr = pa.array(["foo", "bar", "foo foo", "bar bar"])
    seeds = pa.array([2, 3, 4, 5], pa.uint32())
    return input_arr, seeds


@pytest.mark.parametrize("width", [5, 12])
def test_minhash(input_data, width):
    input_arr, seeds = input_data
    result = plc.nvtext.minhash.minhash(
        plc.interop.from_arrow(input_arr), plc.interop.from_arrow(seeds), width
    )
    pa_result = plc.interop.to_arrow(result)
    assert all(len(got) == len(seeds) for got, s in zip(pa_result, input_arr))
    assert pa_result.type == pa.list_(
        pa.field("element", pa.uint32(), nullable=False)
    )
