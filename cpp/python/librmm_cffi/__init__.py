# Copyright (c) 2018, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import

import os
import sys
import atexit
from itertools import chain
from .wrapper import _RMMWrapper
from .wrapper import RMMError     # noqa: F401      # re-exported

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
            lib_path = os.path.join(sys.prefix, 'lib')
            for sys_path in chain([lib_path], sys.path):
                lib = os.path.join(sys_path, path)
                if os.path.isfile(lib):
                    return lib
            return path

    librmm_api = ffi.dlopen(_get_lib_name())
    librmm = _RMMWrapper(ffi, librmm_api)

    # initialize memory manager and register an exit handler to finalize it
    librmm.initialize()
    atexit.register(librmm.finalize)

    del _RMMWrapper
