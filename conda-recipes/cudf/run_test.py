import sys
import os
import pytest
from numba import cuda

if __name__ == '__main__':
    if cuda.is_available():
        exitcode = pytest.main('--pyargs cudf.tests'.split())
    else:
        print("CUDA GPU unavailable")
        exitcode = 0 if os.environ['CUDF_BUILD_NO_GPU_TEST'] else 1

    sys.exit(exitcode)
