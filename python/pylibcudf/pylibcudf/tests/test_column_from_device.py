# Copyright (c) 2024, NVIDIA CORPORATION.

import pyarrow as pa
import pytest
from utils import assert_column_eq

import cudf
from cudf._lib import pylibcudf as plc

VALID_TYPES = [
    pa.int8(),
    pa.int16(),
    pa.int32(),
    pa.int64(),
    pa.uint8(),
    pa.uint16(),
    pa.uint32(),
    pa.uint64(),
    pa.float32(),
    pa.float64(),
    pa.bool_(),
    pa.timestamp("s"),
    pa.timestamp("ms"),
    pa.timestamp("us"),
    pa.timestamp("ns"),
    pa.duration("s"),
    pa.duration("ms"),
    pa.duration("us"),
    pa.duration("ns"),
]


@pytest.fixture(params=VALID_TYPES, ids=repr)
def valid_type(request):
    return request.param


@pytest.fixture
def valid_column(valid_type):
    if valid_type == pa.bool_():
        return pa.array([True, False, True], type=valid_type)
    return pa.array([1, 2, 3], type=valid_type)


def test_from_cuda_array_interface(valid_column):
    col = plc.column.Column.from_cuda_array_interface_obj(
        cudf.Series(valid_column)
    )
    expect = valid_column

    assert_column_eq(expect, col)
