# Copyright (c) 2024, NVIDIA CORPORATION.
# Tell ruff it's OK that some imports occur after the sys.path.insert
# ruff: noqa: E402
import os
import sys

import pyarrow as pa
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "common"))

from utils import DEFAULT_STRUCT_TESTING_TYPE

import cudf._lib.pylibcudf as plc


# This fixture defines the standard set of types that all tests should default to
# running on. If there is a need for some tests to run on a different set of types, that
# type list fixture should also be defined below here if it is likely to be reused
# across modules. Otherwise it may be defined on a per-module basis.
@pytest.fixture(
    scope="session",
    params=[
        pa.int64(),
        pa.float64(),
        pa.string(),
        pa.bool_(),
        pa.list_(pa.int64()),
        DEFAULT_STRUCT_TESTING_TYPE,
    ],
)
def pa_type(request):
    return request.param


# TODO: Test nullable data
@pytest.fixture(scope="session")
def pa_input_column(pa_type):
    if pa.types.is_integer(pa_type) or pa.types.is_floating(pa_type):
        return pa.array([1, 2, 3], type=pa_type)
    elif pa.types.is_string(pa_type):
        return pa.array(["a", "b", "c"], type=pa_type)
    elif pa.types.is_boolean(pa_type):
        return pa.array([True, True, False], type=pa_type)
    elif pa.types.is_list(pa_type):
        # TODO: Add heterogenous sizes
        return pa.array([[1], [2], [3]], type=pa_type)
    elif pa.types.is_struct(pa_type):
        return pa.array([{"v": 1}, {"v": 2}, {"v": 3}], type=pa_type)
    raise ValueError("Unsupported type")


@pytest.fixture(scope="session")
def input_column(pa_input_column):
    return plc.interop.from_arrow(pa_input_column)
