import os
import pathlib

import pytest

import rmm  # noqa: F401


@pytest.fixture(scope="session")
def datadir():
    return pathlib.Path(__file__).parent / "data"


@pytest.hookimpl(tryfirst=True, hookwrapper=True)
def pytest_runtest_makereport(item, call):
    # execute all other hooks to obtain the report object

    outcome = yield
    rep = outcome.get_result()

    # we only look at actual failing test calls, not setup/teardown
    if rep.when == "call" and rep.failed:
        mode = "a" if os.path.exists("failures") else "w"
        with open("failures", mode) as f:
            f.write(rep.longreprtext + "\n")
