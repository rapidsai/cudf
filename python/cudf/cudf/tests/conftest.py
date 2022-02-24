# Copyright (c) 2019-2022, NVIDIA CORPORATION.

import pathlib

import cupy as cp
import numpy as np
import pytest

import rmm  # noqa: F401

from cudf.testing._utils import assert_eq


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


@pytest.fixture(params=[None, "ones"])
def make_weights(request, random_state_tuple):
    """Specific to `test_dataframe_sample*` and `test_series_sample*` tests.
    Only testing weights array that matches type with random state.
    """
    _, gd_random_state, _ = random_state_tuple

    if request.param is None:
        return lambda _: (None, None)
    else:

        def wrapped(size):
            # Uniform distribution, non-normalized
            if isinstance(gd_random_state, np.random.RandomState):
                return np.ones(size), np.ones(size)
            else:
                return np.ones(size), cp.ones(size)

        return wrapped
