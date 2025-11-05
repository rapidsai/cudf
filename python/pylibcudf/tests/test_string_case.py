# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

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
    plc_col = plc.Column.from_arrow(string_col)
    got = plc.strings.case.to_upper(plc_col)
    expect = pc.utf8_upper(string_col)
    assert_column_eq(expect, got)


def test_to_lower(string_col):
    plc_col = plc.Column.from_arrow(string_col)
    got = plc.strings.case.to_lower(plc_col)
    expect = pc.utf8_lower(string_col)
    assert_column_eq(expect, got)


def test_swapcase(string_col):
    plc_col = plc.Column.from_arrow(string_col)
    got = plc.strings.case.swapcase(plc_col)
    expect = pc.utf8_swapcase(string_col)
    assert_column_eq(expect, got)
