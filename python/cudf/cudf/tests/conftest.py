# SPDX-FileCopyrightText: Copyright (c) 2019-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

import itertools
import math
import operator
import os
import pathlib
import zoneinfo

import cupy as cp
import numpy as np
import pandas as pd
import pytest

import rmm  # noqa: F401

import cudf

_CURRENT_DIRECTORY = str(pathlib.Path(__file__).resolve().parent)


@pytest.fixture(scope="session")
def datadir():
    return pathlib.Path(__file__).parent / "data"


# To set and remove the NO_EXTERNAL_ONLY_APIS environment variable we must use
# the sessionstart and sessionfinish hooks rather than a simple autouse,
# session-scope fixture because we need to set these variable before collection
# occurs because the environment variable will be checked as soon as cudf is
# imported anywhere.
def pytest_sessionstart(session):
    """
    Called after the Session object has been created and
    before performing collection and entering the run test loop.
    """
    os.environ["NO_EXTERNAL_ONLY_APIS"] = "1"
    os.environ["_CUDF_TEST_ROOT"] = _CURRENT_DIRECTORY


def pytest_sessionfinish(session, exitstatus):
    """
    Called after whole test run finished, right before
    returning the exit status to the system.
    """
    try:
        del os.environ["NO_EXTERNAL_ONLY_APIS"]
        del os.environ["_CUDF_TEST_ROOT"]
    except KeyError:
        pass


@pytest.fixture(params=[32, 64])
def default_integer_bitwidth(request):
    with cudf.option_context("default_integer_bitwidth", request.param):
        yield request.param


@pytest.fixture(params=[32, 64])
def default_float_bitwidth(request):
    with cudf.option_context("default_float_bitwidth", request.param):
        yield request.param


@pytest.hookimpl(tryfirst=True, hookwrapper=True)
def pytest_runtest_makereport(item, call):
    """Hook to make result information available in fixtures

    This makes it possible for a pytest.fixture to access the current test
    state through `request.node.report`.
    See the `manager` fixture in `test_spilling.py` for an example.

    Pytest doc: <https://docs.pytest.org/en/latest/example/simple.html>
    """
    outcome = yield
    rep = outcome.get_result()

    # Set a report attribute for each phase of a call, which can
    # be "setup", "call", "teardown"
    setattr(item, "report", {rep.when: rep})


def _get_all_zones():
    zones = []
    for zone in zoneinfo.available_timezones():
        # TODO: pandas 3.0 defaults to zoneinfo,
        # so all_zone_names can use zoneinfo.available_timezones()
        try:
            pd.DatetimeTZDtype("ns", zone)
        except KeyError:
            continue
        else:
            zones.append(zone)
    return sorted(zones)


# NOTE: _get_all_zones is a very large list; we likely do NOT want to
# use it for more than a handful of tests
@pytest.fixture(params=_get_all_zones())
def all_timezones(request):
    return request.param


@pytest.fixture(
    params=["America/New_York", "Asia/Tokyo", "CET", "Etc/GMT+1", "UTC"]
)
def limited_timezones(request):
    """
    Small representative set of timezones for testing.
    """
    return request.param


@pytest.fixture(
    params=[
        {
            "LIBCUDF_HOST_DECOMPRESSION": "OFF",
            "LIBCUDF_NVCOMP_POLICY": "ALWAYS",
        },
        {"LIBCUDF_HOST_DECOMPRESSION": "OFF", "LIBCUDF_NVCOMP_POLICY": "OFF"},
        {"LIBCUDF_HOST_DECOMPRESSION": "ON"},
    ],
)
def set_decomp_env_vars(monkeypatch, request):
    env_vars = request.param
    with monkeypatch.context() as m:
        for key, value in env_vars.items():
            m.setenv(key, value)
        yield


arithmetic_ops = [
    operator.add,
    operator.sub,
    operator.mul,
    operator.floordiv,
    operator.truediv,
    operator.mod,
    operator.pow,
]
comparison_ops = [
    operator.eq,
    operator.ne,
    operator.lt,
    operator.le,
    operator.gt,
    operator.ge,
]
bitwise_ops = [
    operator.and_,
    operator.or_,
    operator.xor,
]
unary_ops = [
    math.acos,
    math.acosh,
    math.asin,
    math.asinh,
    math.atan,
    math.atanh,
    math.ceil,
    math.cos,
    math.degrees,
    math.erf,
    math.erfc,
    math.exp,
    math.expm1,
    math.fabs,
    math.floor,
    math.gamma,
    math.lgamma,
    math.log,
    math.log10,
    math.log1p,
    math.log2,
    math.radians,
    math.sin,
    math.sinh,
    math.sqrt,
    math.tan,
    math.tanh,
    operator.pos,
    operator.neg,
    operator.not_,
    operator.invert,
]


@pytest.fixture(params=arithmetic_ops)
def arithmetic_op(request):
    return request.param


@pytest.fixture(
    params=itertools.chain.from_iterable(
        (op.__name__, f"r{op.__name__}") for op in arithmetic_ops
    )
)
def arithmetic_op_method(request):
    """Arithmetic methods defined on Series/DataFrame"""
    return request.param


@pytest.fixture(params=comparison_ops)
def comparison_op(request):
    return request.param


@pytest.fixture
def comparison_op_method(comparison_op):
    """Comparison methods defined on Series/DataFrame"""
    return comparison_op.__name__


@pytest.fixture(params=bitwise_ops)
def bitwise_op(request):
    return request.param


@pytest.fixture(params=unary_ops)
def unary_op(request):
    return request.param


@pytest.fixture(params=arithmetic_ops + comparison_ops)
def binary_op(request):
    return request.param


@pytest.fixture(
    params=itertools.chain(
        itertools.chain.from_iterable(
            (op.__name__, f"r{op.__name__}") for op in arithmetic_ops
        ),
        (op.__name__ for op in comparison_ops),
    )
)
def binary_op_method(request):
    """Binary methods defined on Series/DataFrame"""
    return request.param


@pytest.fixture(
    params=[
        "min",
        "max",
        "sum",
        "product",
        "quantile",
        "all",
        "any",
        "std",
        "var",
        "median",
        "kurtosis",
        "skew",
    ]
)
def reduction_methods(request):
    return request.param


@pytest.fixture(params=["linear", "lower", "higher", "midpoint", "nearest"])
def quantile_interpolation(request):
    return request.param


@pytest.fixture(params=["spearman", "pearson"])
def corr_method(request):
    return request.param


signed_integer_types = ["int8", "int16", "int32", "int64"]
unsigned_integer_types = ["uint8", "uint16", "uint32", "uint64"]
float_types = ["float32", "float64"]
datetime_types = [
    "datetime64[ns]",
    "datetime64[us]",
    "datetime64[ms]",
    "datetime64[s]",
]
timedelta_types = [
    "timedelta64[ns]",
    "timedelta64[us]",
    "timedelta64[ms]",
    "timedelta64[s]",
]
string_types = ["str"]
bool_types = ["bool"]
category_types = ["category"]


@pytest.fixture(params=signed_integer_types)
def signed_integer_types_as_str(request):
    """
    - "int8", "int16", "int32", "int64"
    """
    return request.param


@pytest.fixture(params=unsigned_integer_types)
def unsigned_integer_types_as_str(request):
    """
    - "uint8", "uint16", "uint32", "uint64"
    """
    return request.param


@pytest.fixture(params=signed_integer_types + unsigned_integer_types)
def integer_types_as_str(request):
    """
    - "int8", "int16", "int32", "int64"
    - "uint8", "uint16", "uint32", "uint64"
    """
    return request.param


@pytest.fixture
def integer_types_as_str2(integer_types_as_str):
    """Used for testing cartesian product of integer_types_as_str"""
    return integer_types_as_str


@pytest.fixture(params=float_types)
def float_types_as_str(request):
    """
    - "float32", "float64"
    """
    return request.param


@pytest.fixture(
    params=signed_integer_types + unsigned_integer_types + float_types
)
def numeric_types_as_str(request):
    """
    - "int8", "int16", "int32", "int64"
    - "uint8", "uint16", "uint32", "uint64"
    - "float32", "float64"
    """
    return request.param


@pytest.fixture
def numeric_types_as_str2(numeric_types_as_str):
    """Used for testing cartesian product of numeric_types_as_str"""
    return numeric_types_as_str


@pytest.fixture(
    params=signed_integer_types
    + unsigned_integer_types
    + float_types
    + bool_types
)
def numeric_and_bool_types_as_str(request):
    """
    - "int8", "int16", "int32", "int64"
    - "uint8", "uint16", "uint32", "uint64"
    - "float32", "float64"
    - "bool"
    """
    return request.param


@pytest.fixture(params=datetime_types)
def datetime_types_as_str(request):
    """
    - "datetime64[ns]", "datetime64[us]", "datetime64[ms]", "datetime64[s]"
    """
    return request.param


@pytest.fixture
def datetime_types_as_str2(datetime_types_as_str):
    """Used for testing cartesian product of datetime_types_as_str"""
    return datetime_types_as_str


@pytest.fixture(params=timedelta_types)
def timedelta_types_as_str(request):
    """
    - "timedelta64[ns]", "timedelta64[us]", "timedelta64[ms]", "timedelta64[s]"
    """
    return request.param


@pytest.fixture(params=datetime_types + timedelta_types)
def temporal_types_as_str(request):
    """
    - "datetime64[ns]", "datetime64[us]", "datetime64[ms]", "datetime64[s]"
    - "timedelta64[ns]", "timedelta64[us]", "timedelta64[ms]", "timedelta64[s]"
    """
    return request.param


@pytest.fixture(
    params=signed_integer_types
    + unsigned_integer_types
    + float_types
    + bool_types
    + datetime_types
    + timedelta_types
)
def numeric_and_temporal_types_as_str(request):
    """
    - "int8", "int16", "int32", "int64"
    - "uint8", "uint16", "uint32", "uint64"
    - "float32", "float64"
    - "bool"
    - "datetime64[ns]", "datetime64[us]", "datetime64[ms]", "datetime64[s]"
    - "timedelta64[ns]", "timedelta64[us]", "timedelta64[ms]", "timedelta64[s]"
    """
    return request.param


@pytest.fixture
def numeric_and_temporal_types_as_str2(numeric_and_temporal_types_as_str):
    """Used for testing cartesian product of numeric_and_temporal_types_as_str"""
    return numeric_and_temporal_types_as_str


@pytest.fixture(
    params=signed_integer_types
    + unsigned_integer_types
    + float_types
    + datetime_types
    + timedelta_types
    + string_types
    + bool_types
    + category_types
)
def all_supported_types_as_str(request):
    """
    - "int8", "int16", "int32", "int64"
    - "uint8", "uint16", "uint32", "uint64"
    - "float32", "float64"
    - "datetime64[ns]", "datetime64[us]", "datetime64[ms]", "datetime64[s]"
    - "timedelta64[ns]", "timedelta64[us]", "timedelta64[ms]", "timedelta64[s]"
    - "str"
    - "category"
    - "bool"
    """
    return request.param


@pytest.fixture(params=[list, np.array])
def one_dimensional_array_types(request):
    """1D array containers commonly accepted by cuDF and pandas"""
    return request.param


# pandas can raise warnings for some inputs to the following ufuncs:
numpy_ufuncs = []
for name in dir(np):
    func = getattr(np, name)
    if isinstance(func, np.ufunc) and hasattr(cp, name):
        if func in {
            np.arccos,
            np.arccosh,
            np.arcsin,
            np.arctanh,
            np.fmod,
            np.log,
            np.log10,
            np.log2,
            np.reciprocal,
        }:
            marks = [
                pytest.mark.filterwarnings(
                    "ignore:invalid value encountered:RuntimeWarning"
                ),
                pytest.mark.filterwarnings(
                    "ignore:divide by zero:RuntimeWarning"
                ),
            ]
            numpy_ufuncs.append(pytest.param(func, marks=marks))
        elif func in {
            np.bitwise_and,
            np.bitwise_or,
            np.bitwise_xor,
        }:
            marks = [
                pytest.mark.filterwarnings(
                    "ignore:Operation between non boolean Series:FutureWarning"
                ),
                pytest.mark.filterwarnings(
                    "ignore:Operation between Series with different indexes that are not of numpy boolean:FutureWarning"
                ),
            ]
            numpy_ufuncs.append(pytest.param(func, marks=marks))
        else:
            numpy_ufuncs.append(func)


@pytest.fixture(params=numpy_ufuncs)
def numpy_ufunc(request):
    """Numpy ufuncs also supported by cupy."""
    return request.param


@pytest.fixture(params=[True, False])
def copy(request):
    """Param for `copy` argument"""
    return request.param


@pytest.fixture(params=[True, False])
def deep(request):
    """Param for `deep` argument"""
    return request.param


@pytest.fixture(params=[True, False])
def dropna(request):
    """Param for `dropna` argument"""
    return request.param


@pytest.fixture(params=[True, False])
def skipna(request):
    """Param for `skipna` argument"""
    return request.param


@pytest.fixture(params=[True, False, None])
def nan_as_null(request):
    """Param for `nan_as_null` argument"""
    return request.param


@pytest.fixture(params=[True, False])
def inplace(request):
    """Param for `inplace` argument"""
    return request.param


@pytest.fixture(params=[True, False])
def drop(request):
    """Param for `drop` argument"""
    return request.param


@pytest.fixture(params=[True, False])
def ignore_index(request):
    """Param for `ignore_index` argument"""
    return request.param


@pytest.fixture(params=[True, False])
def ascending(request):
    """Param for `ascending` argument"""
    return request.param


axis_0s = [0, "index"]
axis_1s = [1, "columns"]


@pytest.fixture(params=axis_0s)
def axis_0(request):
    """Param for `axis=0` argument"""
    return request.param


@pytest.fixture(params=axis_1s)
def axis_1(request):
    """Param for `axis=1` argument"""
    return request.param


@pytest.fixture(params=axis_0s + axis_1s)
def axis(request):
    """Param for `axis` argument"""
    return request.param


@pytest.fixture(params=[True, False])
def sort(request):
    """Param for `sort` argument"""
    return request.param


@pytest.fixture(params=[True, False])
def numeric_only(request):
    """Param for `numeric_only` argument"""
    return request.param


@pytest.fixture(params=[True, False, None])
def categorical_ordered(request):
    """Param for `ordered` argument for categorical types"""
    return request.param


@pytest.fixture(params=["left", "right", "both", "neither"])
def interval_closed(request):
    """Param for `closed` argument for interval types"""
    return request.param


@pytest.fixture(params=["all", "any"])
def dropna_how(request):
    """Param for `how` argument"""
    return request.param
