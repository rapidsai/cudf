import pathlib

import pytest


@pytest.fixture(scope="session")
def datadir():
    return pathlib.Path(__file__).parent / "data"
