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

"""Configuration module for librmm, the RAPIDS Memory Manager python interface.
 
   Usage example:
    from librmm_cffi import librmm_config as rmm_cfg
    rmm_cf.use_pool_allocator = True
    rmm_cfg.initial_pool_size = 2 ** 30 # one GB
    import cudf # When cuDF initializes RMM, these settings will be used
"""

# Whether to use a pool allocation strategy. False means to use default cudaMalloc
use_pool_allocator = False

# When `use_pool_allocator` is true, this indicates the initial pool size.
# Zero is used to indicate the default size, which currently is 1/2 total GPU 
# memory.
initial_pool_size = 0

# enable run-time logging of all memory events (alloc, free, realloc)
enable_logging = False