import pytest

import numpy as np
from numba import cuda

from libgdf_cffi import ffi, libgdf, GDFError

from .utils import new_column, unwrap_devary, get_dtype, gen_rand, fix_zeros


def test_cuda_error():
    dtype = np.float32

    col = new_column()
    gdf_dtype = get_dtype(dtype)

    libgdf.gdf_column_view(col, ffi.NULL, ffi.NULL, 0, gdf_dtype)

    #with pytest.raises(GDFError) as raises:
    #    libgdf.gdf_add_generic(col, col, col)

    #raises.match("CUDA ERROR.")
