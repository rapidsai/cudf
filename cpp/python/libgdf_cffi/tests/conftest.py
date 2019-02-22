"""
py.test will automatically detect this file.
"""
import random
import pytest
from libgdf_cffi.tests import utils

from librmm_cffi import librmm_config as rmm_cfg
rmm_cfg.use_pool_allocator = True
from librmm_cffi import librmm as rmm  # noqa: F401, F402 # Ignore flake8 here


# Setup a fixture that is executed once for every test session to set
# a constant seed for the RNG.
@pytest.fixture(scope='session', autouse=True)
def rand_seed():
    # To check whether this is applied.
    # Run with: `py.test -s` to see ensure the following message is printed.
    print("Seeding np.random")
    utils.seed_rand()
    random.seed(0)

# Set up a fixture for the RMM memory manager to initialize and finalize it
# before and after tests.
# @pytest.fixture(scope="session", autouse=True)
# def rmm():
#     librmm.initialize()
#     yield librmm
#     librmm.finalize()
