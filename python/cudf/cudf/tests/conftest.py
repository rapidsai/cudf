import pathlib

import pytest

import rmm  # noqa: F401


@pytest.fixture(scope="session")
def datadir():
    return pathlib.Path(__file__).parent / "data"
