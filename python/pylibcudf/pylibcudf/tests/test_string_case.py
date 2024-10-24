# Copyright (c) 2024, NVIDIA CORPORATION.

import pyarrow as pa
import pyarrow.compute as pc
import pytest
from utils import assert_column_eq

import pylibcudf as plc


@pytest.fixture(scope="module")
def string_col():
    return pa.array(
        ["AbC", "de", "FGHI", "j", "kLm", "nOPq", None, "RsT", None, "uVw"]
    )


def test_to_upper(string_col):
    plc_col = plc.interop.from_arrow(string_col)
    got = plc.strings.case.to_upper(plc_col)
    expected = pc.utf8_upper(string_col)
    assert_column_eq(expected, got)


def test_to_lower(string_col):
    plc_col = plc.interop.from_arrow(string_col)
    got = plc.strings.case.to_lower(plc_col)
    expected = pc.utf8_lower(string_col)
    assert_column_eq(expected, got)


def test_swapcase(string_col):
    plc_col = plc.interop.from_arrow(string_col)
    got = plc.strings.case.swapcase(plc_col)
    expected = pc.utf8_swapcase(string_col)
    assert_column_eq(expected, got)
