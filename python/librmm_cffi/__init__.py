from __future__ import absolute_import

import os
import sys
import atexit

from .wrapper import _librmm_wrapper
from .wrapper import RMMError           # re-exported

try:
    from .librmm_cffi import ffi
except ImportError:
    pass
else:
    def _get_lib_name():
        if os.name == 'posix':
            # TODO this will need to be changed when packaged for distribution
            if sys.platform == 'darwin':
                path = 'librmm.dylib'
            else:
                path = 'librmm.so'
        else:
            raise NotImplementedError('OS {} not supported'.format(os.name))
        # Prefer local version of the library if it exists
        localpath = os.path.join('.', path)
        if os.path.isfile(localpath):
            return localpath
        else:
            return path

    librmm_api = ffi.dlopen(_get_lib_name())
    librmm = _librmm_wrapper(ffi, librmm_api)

    # initialize memory manager and register an exit handler to finalize it
    librmm.initialize()
    atexit.register(librmm.finalize)

    del _librmm_wrapper

