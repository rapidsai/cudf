import pytest

from librmm_cffi import librmm


# Set up a fixture for the RMM memory manager to initialize and finalize it
# before and after tests.
@pytest.fixture(scope="session", autouse=True)
def rmm():
    librmm.initialize()
    yield librmm
    librmm.finalize()
