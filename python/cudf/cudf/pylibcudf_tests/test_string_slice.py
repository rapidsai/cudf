# Copyright (c) 2024, NVIDIA CORPORATION.

import pyarrow as pa
import pytest
from utils import assert_column_eq

import cudf._lib.pylibcudf as plc


@pytest.fixture(scope="module")
def string_col():
    return pa.array(["AbC"])


@pytest.fixture(
    scope="module",
    params=[
        (1, 3, 1),
    ],
)
def pa_start_stop_step(request):
    return (
        pa.scalar(request.param[0], type=pa.int32()),
        pa.scalar(request.param[1], type=pa.int32()),
        pa.scalar(request.param[2], type=pa.int32()),
    )


@pytest.fixture(scope="module")
def plc_start_stop_step(pa_start_stop_step):
    return (
        plc.interop.from_arrow(pa_start_stop_step[0]),
        plc.interop.from_arrow(pa_start_stop_step[1]),
        plc.interop.from_arrow(pa_start_stop_step[2]),
    )


def test_slice(string_col, pa_start_stop_step, plc_start_stop_step):
    plc_col = plc.interop.from_arrow(string_col)

    pa_start, pa_stop, pa_step = pa_start_stop_step
    plc_start, plc_stop, plc_step = plc_start_stop_step

    expected = pa.compute.utf8_slice_codeunits(
        string_col, pa_start.as_py(), pa_stop.as_py(), pa_step.as_py()
    )
    got = plc.strings.slice.slice_strings(
        plc_col, start=plc_start, stop=plc_stop, step=plc_step
    )

    assert_column_eq(expected, got)
