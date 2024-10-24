# Copyright (c) 2024, NVIDIA CORPORATION.

import pyarrow as pa
import pytest
from utils import assert_column_eq

import pylibcudf as plc

data_strings = [
    "AbC",
    "123abc",
    "",
    " ",
    None,
    "aAaaaAAaa",
    " ab c ",
    "abc123",
    "    ",
    "\tabc\t",
    "\nabc\n",
    "\r\nabc\r\n",
    "\t\n abc \n\t",
    "!@#$%^&*()",
    "   abc!!!   ",
    "   abc\t\n!!!   ",
    "__abc__",
    "abc\n\n",
    "123abc456",
    "abcxyzabc",
]

strip_chars = [
    "a",
    "",
    " ",
    "\t",
    "\n",
    "\r\n",
    "!",
    "@#",
    "123",
    "xyz",
    "abc",
    "__",
    " \t\n",
    "abc123",
]


@pytest.fixture
def pa_col():
    return pa.array(data_strings, type=pa.string())


@pytest.fixture
def plc_col(pa_col):
    return plc.interop.from_arrow(pa_col)


@pytest.fixture(params=strip_chars)
def pa_char(request):
    return pa.scalar(request.param, type=pa.string())


@pytest.fixture
def plc_char(pa_char):
    return plc.interop.from_arrow(pa_char)


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


def test_strip_right(pa_col, plc_col, pa_char, plc_char):
    def strip_string(st, char):
        if st is None:
            return None

        elif char == "":
            return st.rstrip()
        return st.rstrip(char)

    expected = pa.array(
        [strip_string(x, pa_char.as_py()) for x in pa_col.to_pylist()],
        type=pa.string(),
    )

    got = plc.strings.strip.strip(
        plc_col, plc.strings.SideType.RIGHT, plc_char
    )
    assert_column_eq(expected, got)


def test_strip_left(pa_col, plc_col, pa_char, plc_char):
    def strip_string(st, char):
        if st is None:
            return None

        elif char == "":
            return st.lstrip()
        return st.lstrip(char)

    expected = pa.array(
        [strip_string(x, pa_char.as_py()) for x in pa_col.to_pylist()],
        type=pa.string(),
    )

    got = plc.strings.strip.strip(plc_col, plc.strings.SideType.LEFT, plc_char)
    assert_column_eq(expected, got)
