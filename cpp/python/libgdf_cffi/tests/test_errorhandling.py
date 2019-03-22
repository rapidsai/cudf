import numpy as np

from libgdf_cffi import ffi, libgdf  # , GDFError

from libgdf_cffi.tests.utils import new_column, get_dtype


def test_cuda_error():
    dtype = np.float32

    col = new_column()
    gdf_dtype = get_dtype(dtype)

    libgdf.gdf_column_view(col, ffi.NULL, ffi.NULL, 0, gdf_dtype)

    # with pytest.raises(GDFError) as raises:
    #     libgdf.gdf_add_generic(col, col, col)

    # raises.match("CUDA ERROR.")
