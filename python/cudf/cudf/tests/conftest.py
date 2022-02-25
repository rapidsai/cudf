# Copyright (c) 2019-2022, NVIDIA CORPORATION.

import os
import pathlib

import cupy as cp
import numpy as np
import pytest

import rmm  # noqa: F401

from cudf.testing._utils import assert_eq

_CURRENT_DIRECTORY = str(pathlib.Path(__file__).resolve().parent)


@pytest.fixture(scope="session")
def datadir():
    return pathlib.Path(__file__).parent / "data"


@pytest.fixture(
    params=[None, 42, np.random.RandomState, cp.random.RandomState]
)
def random_state_tuple(request):
    """
    Specific to `test_dataframe_sample*` and `test_series_sample*` tests.
    A pytest fixture of valid `random_state` parameter pairs for pandas
    and cudf. Valid parameter combinations, and what to check for each pair
    are listed below:

    pandas:   None,   seed(int),  np.random.RandomState,  np.random.RandomState
    cudf:     None,   seed(int),  np.random.RandomState,  cp.random.RandomState
    ------
    check:    shape,  shape,      exact result,           shape

    Each column above stands for one valid parameter combination and check.
    """

    def shape_checker(expected, got):
        assert expected.shape == got.shape

    def exact_checker(expected, got):
        assert_eq(expected, got)

    seed_or_state_ctor = request.param
    if seed_or_state_ctor is None:
        return None, None, shape_checker
    elif isinstance(seed_or_state_ctor, int):
        return seed_or_state_ctor, seed_or_state_ctor, shape_checker
    elif seed_or_state_ctor == np.random.RandomState:
        return seed_or_state_ctor(42), seed_or_state_ctor(42), exact_checker
    elif seed_or_state_ctor == cp.random.RandomState:
        return np.random.RandomState(42), seed_or_state_ctor(42), shape_checker
    else:
        pytest.skip("Unsupported params.")


@pytest.fixture(params=[None, "builtin-list", "nd-arrays"])
def make_weights(request, random_state_tuple):
    """Specific to `test_dataframe_sample*` and `test_series_sample*` tests.
    Only testing weights array that matches type with random state.
    """
    _, gd_random_state, _ = random_state_tuple

    if request.param is None:
        return lambda _: (None, None)
    elif request.param == "builtin-list":
        return lambda size: ([1] * size, [1] * size)
    else:

        def wrapped(size):
            # Uniform distribution, non-normalized
            if isinstance(gd_random_state, np.random.RandomState):
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
