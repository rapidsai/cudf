import os
import pathlib

import pytest

import rmm  # noqa: F401


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


def pytest_sessionfinish(session, exitstatus):
    """
    Called after whole test run finished, right before
    returning the exit status to the system.
    """
    del os.environ["NO_EXTERNAL_ONLY_APIS"]
