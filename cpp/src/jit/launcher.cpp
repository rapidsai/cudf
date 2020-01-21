/*
 * Copyright (c) 2019, NVIDIA CORPORATION.
 *
 * Copyright 2018-2019 BlazingDB, Inc.
 *     Copyright 2018 Christian Noboa Mardini <christian@blazingdb.com>
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <jit/launcher.h>
#include <jit/parser.h>
#include <cstdint>
#include <chrono>

namespace cudf {
namespace jit {

    launcher::launcher(
      const std::string& hash,
      const std::string& cuda_source,
      const std::vector<std::string>& header_names,
      const std::vector<std::string>& compiler_flags,
      jitify::experimental::file_callback_type file_callback,
      cudaStream_t stream
    )
     : cache_instance{cudf::jit::cudfJitCache::Instance()}
     , stream(stream) 
    {
      program = cache_instance.getProgram(
                  hash,
                  cuda_source.c_str(),
                  header_names,
                  compiler_flags,
                  file_callback
                );
    }

    launcher::launcher(launcher&& launcher)
     : program {std::move(launcher.program)}
     , cache_instance {cudf::jit::cudfJitCache::Instance()}
     , kernel_inst {std::move(launcher.kernel_inst)}
     , stream {launcher.stream}
    { }

} // namespace jit
} // namespace cudf
