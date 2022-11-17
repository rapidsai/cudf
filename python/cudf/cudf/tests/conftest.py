# Copyright (c) 2019-2022, NVIDIA CORPORATION.

import itertools
import os
import pathlib

import cupy as cp
import numpy as np
import pytest

import rmm  # noqa: F401

import cudf
from cudf.testing._utils import assert_eq

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
    old_default = cudf.get_option("default_integer_bitwidth")
    cudf.set_option("default_integer_bitwidth", request.param)
    yield request.param
    cudf.set_option("default_integer_bitwidth", old_default)


@pytest.fixture(params=[32, 64])
def default_float_bitwidth(request):
    old_default = cudf.get_option("default_float_bitwidth")
    cudf.set_option("default_float_bitwidth", request.param)
    yield request.param
    cudf.set_option("default_float_bitwidth", old_default)


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
