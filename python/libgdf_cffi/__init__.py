from __future__ import absolute_import

from .wrapper import _libgdf_wrapper
from .wrapper import GDFError           # re-exported

try:
    from .libgdf_cffi import ffi
except ImportError:
    pass
else:
    libgdf_api = ffi.dlopen('libgdf.dylib')
    libgdf = _libgdf_wrapper(ffi, libgdf_api)

    del _libgdf_wrapper
