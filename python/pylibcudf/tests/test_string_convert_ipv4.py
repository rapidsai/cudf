# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
import pyarrow as pa
from utils import assert_column_eq

import pylibcudf as plc


def test_ipv4_to_integers():
    got = plc.strings.convert.convert_ipv4.ipv4_to_integers(
        plc.Column.from_arrow(pa.array(["123.45.67.890", None]))
    )
    expect = pa.array([2066564730, None], type=pa.uint32())
    assert_column_eq(expect, got)


def test_integers_to_ipv4():
    got = plc.strings.convert.convert_ipv4.integers_to_ipv4(
        plc.Column.from_arrow(pa.array([1, 0, None], type=pa.uint32()))
    )
    expect = pa.array(["0.0.0.1", "0.0.0.0", None])
    assert_column_eq(expect, got)


def test_is_ipv4():
    got = plc.strings.convert.convert_ipv4.is_ipv4(
        plc.Column.from_arrow(pa.array(["0.0.0.1", "1.2.34", "A", None]))
    )
    expect = pa.array([True, False, False, None])
    assert_column_eq(expect, got)
