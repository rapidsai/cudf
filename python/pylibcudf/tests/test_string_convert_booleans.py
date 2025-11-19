# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

import pyarrow as pa
from utils import assert_column_eq

import pylibcudf as plc


def test_to_booleans():
    pa_array = pa.array(["true", None, "True"])
    got = plc.strings.convert.convert_booleans.to_booleans(
        plc.Column.from_arrow(pa_array),
        plc.Scalar.from_arrow(pa.scalar("True")),
    )
    expect = pa.array([False, None, True])
    assert_column_eq(expect, got)


def test_from_booleans():
    pa_array = pa.array([True, None, False])
    got = plc.strings.convert.convert_booleans.from_booleans(
        plc.Column.from_arrow(pa_array),
        plc.Scalar.from_arrow(pa.scalar("A")),
        plc.Scalar.from_arrow(pa.scalar("B")),
    )
    expect = pa.array(["A", None, "B"])
    assert_column_eq(expect, got)
