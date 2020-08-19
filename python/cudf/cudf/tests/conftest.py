import pathlib

import pandas as pd
import pytest
from packaging.version import parse

import rmm  # noqa: F401


@pytest.fixture(scope="session")
def datadir():
    return pathlib.Path(__file__).parent / "data"


@pytest.fixture(scope="session")
def pd_version_fail():
    if parse("1.0") <= parse(pd.__version__) < parse("1.1"):
        return True
    else:
        return False
