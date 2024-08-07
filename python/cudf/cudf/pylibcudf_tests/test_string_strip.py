# Copyright (c) 2024, NVIDIA CORPORATION.

import pyarrow as pa
import pytest
from utils import assert_column_eq

import cudf._lib.pylibcudf as plc


@pytest.fixture
def pa_col():
    return pa.array(["AbC", "123abc", "", " ", None])


@pytest.fixture
def plc_col(pa_col):
    return plc.interop.from_arrow(pa_col)


@pytest.fixture(params=["a", "", " "])
def pa_char(request):
    return pa.scalar(request.param, type=pa.string())


@pytest.fixture
def plc_char(pa_char):
    return plc.interop.from_arrow(pa_char)


# TODO: add more tests
def test_strip(pa_col, plc_col, pa_char, plc_char):
    expected = pa.compute.utf8_trim(pa_col, pa_char.as_py().encode())
    got = plc.strings.strip.strip(plc_col, plc.strings.SideType.BOTH, plc_char)
    assert_column_eq(expected, got)
