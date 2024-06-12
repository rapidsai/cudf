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
        (0, 3, -1),
    ],
)
def pa_start_stop_step(request):
    return tuple(pa.scalar(x, type=pa.int32()) for x in request.param)


@pytest.fixture(scope="module")
def plc_start_stop_step(pa_start_stop_step):
    return tuple(plc.interop.from_arrow(x) for x in pa_start_stop_step)


def test_slice(string_col, pa_start_stop_step, plc_start_stop_step):
    plc_col = plc.interop.from_arrow(string_col)

    pa_start, pa_stop, pa_step = pa_start_stop_step
    plc_start, plc_stop, plc_step = plc_start_stop_step

    def slice_string(st, start, stop, step):
        return st[start:stop:step] if st is not None else None

    expected = pa.array(
        [
            slice_string(x, pa_start.as_py(), pa_stop.as_py(), pa_step.as_py())
            for x in string_col.to_pylist()
        ],
        type=pa.string(),
    )

    got = plc.strings.slice.slice_strings(
        plc_col, start=plc_start, stop=plc_stop, step=plc_step
    )

    assert_column_eq(expected, got)
