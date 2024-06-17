# Copyright (c) 2024, NVIDIA CORPORATION.
# Tell ruff it's OK that some imports occur after the sys.path.insert
# ruff: noqa: E402
import os
import sys

import pyarrow as pa
import pytest

import cudf._lib.pylibcudf as plc

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "common"))

from utils import DEFAULT_STRUCT_TESTING_TYPE


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


@pytest.fixture(
    scope="session",
    params=[
        pa.int64(),
        pa.float64(),
        pa.uint64(),
    ],
)
def numeric_pa_type(request):
    return request.param


@pytest.fixture(
    scope="session", params=[opt for opt in plc.types.Interpolation]
)
def interp_opt(request):
    return request.param


@pytest.fixture(
    scope="session",
    params=[opt for opt in plc.types.Sorted],
)
def sorted_opt(request):
    return request.param


@pytest.fixture(scope="session", params=[False, True])
def has_nulls(request):
    return request.param
