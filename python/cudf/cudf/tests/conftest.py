import pathlib

import pytest

import rmm


# Set up a fixture for the RMM memory manager to initialize and finalize it
# before and after tests.
@pytest.fixture(scope="session", autouse=True)
def setup_rmm():
    rmm.initialize()
    yield rmm
    rmm.finalize()


@pytest.fixture(scope="session")
def datadir():
    return pathlib.Path(__file__).parent / "data"
