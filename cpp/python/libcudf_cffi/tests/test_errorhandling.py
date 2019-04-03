import numpy as np

from libcudf_cffi import ffi, libcudf  # , GDFError

from libcudf_cffi.tests.utils import new_column, get_dtype


def test_cuda_error():
    dtype = np.float32

    col = new_column()
    gdf_dtype = get_dtype(dtype)

    libcudf.gdf_column_view(col, ffi.NULL, ffi.NULL, 0, gdf_dtype)

    # with pytest.raises(GDFError) as raises:
    #     libcudf.gdf_add_generic(col, col, col)

    # raises.match("CUDA ERROR.")
