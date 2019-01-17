from __future__ import absolute_import

import os
import sys
from itertools import chain
from .wrapper import _libgdf_wrapper
from .wrapper import GDFError         # noqa: F401  # re-exported

try:
    from .libgdf_cffi import ffi
except ImportError:
    pass
else:
    def _get_lib_name():
        if os.name == 'posix':
            # TODO this will need to be changed when packaged for distribution
            if sys.platform == 'darwin':
                path = 'libcudf.dylib'
            else:
                path = 'libcudf.so'
        else:
            raise NotImplementedError('OS {} not supported'.format(os.name))
        # Prefer local version of the library if it exists
        localpath = os.path.join('.', path)
        if os.path.isfile(localpath):
            return localpath
        else:
            lib_path = os.path.join(sys.prefix, 'lib')
            for sys_path in chain([lib_path], sys.path):
                lib = os.path.join(sys_path, path)
                if os.path.isfile(lib):
                    return lib
            return path

    libgdf_api = ffi.dlopen(_get_lib_name())
    libgdf = _libgdf_wrapper(ffi, libgdf_api)

    del _libgdf_wrapper
