from __future__ import print_function, absolute_import, division
from numba import findlib
import ctypes
import os
import platform
import warnings

def library_extension():
    p = platform.system()
    if p == 'Linux':
        return 'so'
    if p == 'Windows':
        return 'dll'
    if p == 'Darwin':
        return 'dylib'

def load_lib(libname):
    fullname = 'accelerate_%s.%s' % (libname, library_extension())
    devlib = os.path.join(os.path.abspath(os.path.dirname(__file__)), fullname)
    if os.path.exists(devlib):
        libpath = devlib
        warnings.warn('Using in-tree library %s' % libpath)
    else:
        libpath = os.path.join(findlib.get_lib_dir(), fullname)

    return ctypes.CDLL(libpath)


