# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

import pyarrow as pa
import pyarrow.compute as pc
import pytest
from utils import assert_column_eq

import pylibcudf as plc


@pytest.fixture
def str_data():
    pa_data = pa.array(["A", None])
    return pa_data, plc.Column.from_arrow(pa_data)


def test_count_characters(str_data):
    got = plc.strings.attributes.count_characters(str_data[1])
    expect = pc.utf8_length(str_data[0])
    assert_column_eq(expect, got)


def test_count_bytes(str_data):
    got = plc.strings.attributes.count_characters(str_data[1])
    expect = pc.binary_length(str_data[0])
    assert_column_eq(expect, got)


def test_code_points(str_data):
    got = plc.strings.attributes.code_points(str_data[1])
    exp_data = [ord(str_data[0].to_pylist()[0])]
    expect = pa.chunked_array([exp_data], type=pa.int32())
    assert_column_eq(expect, got)
