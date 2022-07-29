#
# Copyright (c) 2020-2022, NVIDIA CORPORATION.
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

from setuptools import find_packages
from skbuild import setup
from wheel.bdist_wheel import bdist_wheel as _bdist_wheel

import versioneer
import os


'''
copy this trick from https://github.com/ssciwr/clang-format-wheel/blob/main/setup.py
since the C++ code compiled by this cpp module is not a Python C extension
override the platform to be py3-none
'''
class genericpy_bdist_wheel(_bdist_wheel):
    def finalize_options(self):
        _bdist_wheel.finalize_options(self)
        self.root_is_pure = False

    def get_tag(self):
        python, abi, plat = _bdist_wheel.get_tag(self)
        python, abi = "py3", "none"
        return python, abi, plat


cmdclass = versioneer.get_cmdclass()
cmdclass['bdist_wheel'] = genericpy_bdist_wheel


def exclude_libcxx_symlink(cmake_manifest):
    return list(filter(lambda name: not ('include/rapids/libcxx/include' in name), cmake_manifest))


setup(name='libcudf'+os.getenv("PYTHON_PACKAGE_CUDA_SUFFIX", default=""),
      description="cuDF C++ library",
      version=versioneer.get_version(),
      classifiers=[
        "Intended Audience :: Developers",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9"
      ],
      author="NVIDIA Corporation",
      cmake_process_manifest_hook=exclude_libcxx_symlink,
      packages=find_packages(include=['libcudf']),
      license="Apache",
      cmdclass=cmdclass,
      zip_safe=False
      )
