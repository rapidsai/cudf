import sys
import os
import pytest
from numba import cuda

if cuda.is_available():
    exitcode = pytest.main('-v', 'pygdf')
else:
    print("CUDA GPU unavailable")
    exitcode = 1 if os.environ['PYGDF_BUILD_NO_GPU_TEST'] else 0

sys.exit(exitcode)
