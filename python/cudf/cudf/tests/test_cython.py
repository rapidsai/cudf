# Copyright (c) 2021, NVIDIA CORPORATION.
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

import functools
import importlib
import sys


def py_func(func):
    """
    Wraps func in a plain Python function.
    """

    @functools.wraps(func)
    def wrapped(*args, **kwargs):
        return func(*args, **kwargs)

    return wrapped


cython_test_modules = ["cudf._lib.tests.test_pack_unpack"]


for mod in cython_test_modules:
    try:
        # For each callable in `mod` with name `test_*`,
        # wrap the callable in a plain Python function
        # and set the result as an attribute of this module.
        mod = importlib.import_module(mod)
        for name in dir(mod):
            item = getattr(mod, name)
            if callable(item) and name.startswith("test_"):
                item = py_func(item)
                setattr(sys.modules[__name__], name, item)
    except ImportError:
        pass
