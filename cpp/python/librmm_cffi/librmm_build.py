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

import cffi
import os

header = os.environ.get('RMM_HEADER', '../../src/rmm/memory.h')

ffibuilder = cffi.FFI()
ffibuilder.set_source("librmm_cffi.librmm_cffi", None)

with open(header, 'r') as fin:
    ffibuilder.cdef(fin.read())

if __name__ == "__main__":
    ffibuilder.compile()
