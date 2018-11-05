from __future__ import absolute_import

import os
import sys

from .wrapper import _libgdf_wrapper
from .wrapper import GDFError           # re-exported

try:
    from .libgdf_cffi import ffi
except ImportError:
    pass
else:
    def _get_lib_name():
        if os.name == 'posix':
            # TODO this will need to be changed when packaged for distribution
            if sys.platform == 'darwin':
                path = 'libgdf.dylib'
            else:
                path = 'libgdf.so'
        else:
            raise NotImplementedError('OS {} not supported'.format(os.name))
        # Prefer local version of the library if it exists
        localpath = os.path.join('.', path)
        if os.path.isfile(localpath):
            return localpath
        else:
            return path

    libgdf_api = ffi.dlopen(_get_lib_name())
    libgdf = _libgdf_wrapper(ffi, libgdf_api)

    del _libgdf_wrapper
