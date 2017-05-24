try:
    from .libgdf_cffi import ffi
except ImportError:
    pass
else:
    libgdf = ffi.dlopen('libgdf.so')
