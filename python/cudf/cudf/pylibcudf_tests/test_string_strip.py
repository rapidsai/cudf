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
    def strip_string(st, char):
        if st is None:
            return None

        elif char == "":
            return st.strip()
        return st.strip(char)

    expected = pa.array(
        [strip_string(x, pa_char.as_py()) for x in pa_col.to_pylist()],
        type=pa.string(),
    )

    got = plc.strings.strip.strip(plc_col, plc.strings.SideType.BOTH, plc_char)
    assert_column_eq(expected, got)
