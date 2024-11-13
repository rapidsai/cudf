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

# RTLD_LOCAL is here for safety... using it loads symbols into the library-specific
# table maintained by the loader, but not into the global namespace where they
# may conflict with symbols from other loaded DSOs.
PREFERRED_LOAD_FLAG = ctypes.RTLD_LOCAL


def _load_system_installation(soname: str):
    """Try to dlopen() the library indicated by ``soname``
    Raises ``OSError`` if library cannot be loaded.
    """
    return ctypes.CDLL(soname, PREFERRED_LOAD_FLAG)


def _load_wheel_installation(soname: str):
    """Try to dlopen() the library indicated by ``soname``

    Returns ``None`` if the library cannot be loaded.
    """
    out = None
    for lib_dir in ("lib", "lib64"):
        if os.path.isfile(
            lib := os.path.join(os.path.dirname(__file__), lib_dir, soname)
        ):
            out = ctypes.CDLL(lib, PREFERRED_LOAD_FLAG)
            break
    return out


def load_library():
    """Dynamically load libcudf.so and its dependencies"""
    try:
        # libkvikio must be loaded before libcudf because libcudf references its symbols
        import libkvikio

        libkvikio.load_library()
    except ModuleNotFoundError:
        # libcudf's runtime dependency on libkvikio may be satisfied by a natively
        # installed library or a conda package, in which case the import will fail and
        # we assume the library is discoverable on system paths.
        pass

    prefer_system_installation = (
        os.getenv("RAPIDS_LIBCUDF_PREFER_SYSTEM_LIBRARY", "false").lower()
        != "false"
    )

    soname = "libcudf.so"
    libcudf_lib = None
    if prefer_system_installation:
        # Prefer a system library if one is present to
        # avoid clobbering symbols that other packages might expect, but if no
        # other library is present use the one in the wheel.
        try:
            libcudf_lib = _load_system_installation(soname)
        except OSError:
            libcudf_lib = _load_wheel_installation(soname)
    else:
        # Prefer the libraries bundled in this package. If they aren't found
        # (which might be the case in builds where the library was prebuilt before
        # packaging the wheel), look for a system installation.
        libcudf_lib = _load_wheel_installation(soname)
        if libcudf_lib is None:
            libcudf_lib = _load_system_installation(soname)

    # The caller almost never needs to do anything with this library, but no
    # harm in offering the option since this object at least provides a handle
    # to inspect where libcudf was loaded from.
    return libcudf_lib
