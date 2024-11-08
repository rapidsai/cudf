# Copyright (c) 2024, NVIDIA CORPORATION.

import pyarrow as pa
import pytest
from utils import assert_column_eq

import rmm

import pylibcudf as plc

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


class DataBuffer:
    def __init__(self, obj, dtype):
        self.obj = rmm.DeviceBuffer.to_device(obj)
        self.dtype = dtype
        self.shape = (int(len(self.obj) / self.dtype.itemsize),)
        self.strides = (self.dtype.itemsize,)
        self.typestr = self.dtype.str

    @property
    def __cuda_array_interface__(self):
        return {
            "data": self.obj.__cuda_array_interface__["data"],
            "shape": self.shape,
            "strides": self.strides,
            "typestr": self.typestr,
            "version": 0,
        }


@pytest.fixture
def input_column(valid_type):
    if valid_type == pa.bool_():
        return pa.array([True, False, True], type=valid_type)
    return pa.array([1, 2, 3], type=valid_type)


@pytest.fixture
def iface_obj(input_column):
    data = input_column.to_numpy(zero_copy_only=False)
    return DataBuffer(data.view("uint8"), data.dtype)


def test_from_cuda_array_interface(input_column, iface_obj):
    col = plc.column.Column.from_cuda_array_interface_obj(iface_obj)

    assert_column_eq(input_column, col)
