# Copyright (c) 2024, NVIDIA CORPORATION.
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
#

import ctypes
import os


def load_library():
    # This is loading the libarrow shared library in situations where it comes from the
    # pyarrow package (i.e. when installed as a wheel).
    import pyarrow  # noqa: F401

    # Dynamically load libcudf.so. Prefer a system library if one is present to
    # avoid clobbering symbols that other packages might expect, but if no
    # other library is present use the one in the wheel.
    libcudf_lib = None
    try:
        libcudf_lib = ctypes.CDLL("libcudf.so", ctypes.RTLD_GLOBAL)
    except OSError:
        this_dir = os.path.dirname(__file__)
        lib_dir = os.path.join(this_dir, "lib")
        lib64_dir = os.path.join(this_dir, "lib64")
        for real_lib_dir in (lib_dir, lib64_dir):
            if os.path.isdir(real_lib_dir):
                break
        else:
            real_lib_dir = None

        # If neither directory was found in the wheel, we assume we are in an
        # environment where the C++ library is already installed somewhere else and the
        # CMake build of the libcudf Python package was a no-op.
        if real_lib_dir is not None:
            libcudf_lib = ctypes.CDLL(
                os.path.join(real_lib_dir, "libcudf.so"),
                ctypes.RTLD_GLOBAL,
            )

    # The caller almost never needs to do anything with this library, but no
    # harm in offering the option since this object at least provides a handle
    # to inspect where libcudf was loaded from.
    return libcudf_lib
