# Copyright (c) 2019-2025, NVIDIA CORPORATION.

import itertools
import os
import pathlib

import cupy as cp
import numpy as np
import pytest

import rmm  # noqa: F401

import cudf
from cudf.testing import assert_eq

_CURRENT_DIRECTORY = str(pathlib.Path(__file__).resolve().parent)


@pytest.fixture(scope="session")
def datadir():
    return pathlib.Path(__file__).parent / "data"


@pytest.fixture(
    params=itertools.product([0, 2, None], [0.3, None]),
    ids=lambda arg: f"n={arg[0]}-frac={arg[1]}",
)
def sample_n_frac(request):
    """
    Specific to `test_sample*` tests.
    """
    n, frac = request.param
    if n is not None and frac is not None:
        pytest.skip("Cannot specify both n and frac.")
    return n, frac


def shape_checker(expected, got):
    assert expected.shape == got.shape


def exact_checker(expected, got):
    assert_eq(expected, got)


@pytest.fixture(
    params=[
        (None, None, shape_checker),
        (42, 42, shape_checker),
        (np.random.RandomState(42), np.random.RandomState(42), exact_checker),
    ],
    ids=["None", "IntSeed", "NumpyRandomState"],
)
def random_state_tuple_axis_1(request):
    """
    Specific to `test_sample*_axis_1` tests.
    A pytest fixture of valid `random_state` parameter pairs for pandas
    and cudf. Valid parameter combinations, and what to check for each pair
    are listed below:

    pandas:   None,   seed(int),  np.random.RandomState
    cudf:     None,   seed(int),  np.random.RandomState
    ------
    check:    shape,  shape,      exact result

    Each column above stands for one valid parameter combination and check.
    """

    return request.param


@pytest.fixture(
    params=[
        (None, None, shape_checker),
        (42, 42, shape_checker),
        (np.random.RandomState(42), np.random.RandomState(42), exact_checker),
        (np.random.RandomState(42), cp.random.RandomState(42), shape_checker),
    ],
    ids=["None", "IntSeed", "NumpyRandomState", "CupyRandomState"],
)
def random_state_tuple_axis_0(request):
    """
    Specific to `test_sample*_axis_0` tests.
    A pytest fixture of valid `random_state` parameter pairs for pandas
    and cudf. Valid parameter combinations, and what to check for each pair
    are listed below:

    pandas:   None,   seed(int),  np.random.RandomState,  np.random.RandomState
    cudf:     None,   seed(int),  np.random.RandomState,  cp.random.RandomState
    ------
    check:    shape,  shape,      exact result,           shape

    Each column above stands for one valid parameter combination and check.
    """

    return request.param


@pytest.fixture(params=[None, "builtin_list", "ndarray"])
def make_weights_axis_0(request):
    """Specific to `test_sample*_axis_0` tests.
    Only testing weights array that matches type with random state.
    """

    if request.param is None:
        return lambda *_: (None, None)
    elif request.param == "builtin-list":
        return lambda size, _: ([1] * size, [1] * size)
    else:

        def wrapped(size, numpy_weights_for_cudf):
            # Uniform distribution, non-normalized
            if numpy_weights_for_cudf:
                return np.ones(size), np.ones(size)
            else:
                return np.ones(size), cp.ones(size)

        return wrapped


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


@pytest.fixture(params=[True, False])
def dropna(request):
    """Param for `dropna` argument"""
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
def ignore_index(request):
    """Param for `ignore_index` argument"""
    return request.param


@pytest.fixture(params=[True, False])
def ascending(request):
    """Param for `ascending` argument"""
    return request.param
