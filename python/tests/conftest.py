"""
py.test will automatically detect this file.
"""
import random
import pytest
from . import utils

from librmm_cffi import librmm

# Setup a fixture that is executed once for every test session to set
# a constant seed for the RNG.
@pytest.fixture(scope='session', autouse=True)
def rand_seed():
    # To check whether this is applied.
    # Run with: `py.test -s` to see ensure the following message is printed.
    print("Seeding np.random")
    utils.seed_rand()
    random.seed(0)

# Setup a fixture for the RMM memory manager to initialize and finalize it before
# and after tests.
@pytest.fixture(scope="session", autouse=True)
def rmm():
    yield librmm # librmm is initialized on import, finalized on exit
