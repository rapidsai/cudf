# Copyright (c) 2019-2022, NVIDIA CORPORATION.

import os
import pathlib

import pytest

import rmm  # noqa: F401

_CURRENT_DIRECTORY = os.path.dirname(os.path.realpath(__file__))


@pytest.fixture(scope="session")
def datadir():
    return pathlib.Path(__file__).parent / "data"


# To set and remove the NO_EXTERNAL_ONLY_APIS environment variable we must use
# the sessionstart and sessionfinish hooks rather than a simple autouse,
# session-scope fixture because we need to set these variable before collection
# occurs because the envrionment variable will be checked as soon as cudf is
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
