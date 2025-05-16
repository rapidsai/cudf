/*
 * Copyright (c) 2023-2024, NVIDIA CORPORATION.
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

#include <benchmarks/fixture/nvbench_fixture.hpp>

#include <nvbench/main.cuh>

#include <string>
#include <vector>

namespace cudf {

// strip off the rmm_mode and cuio_host_mem parameters before passing the
// remaining arguments to nvbench::option_parser
void benchmark_arg_handler(std::vector<std::string>& args)
{
  std::vector<std::string> _cudf_tmp_args;

  for (std::size_t i = 0; i < args.size(); ++i) {
    std::string arg = args[i];
    if (arg == cudf::detail::rmm_mode_param) {
      i++;  // skip the next argument
    } else if (arg == cudf::detail::cuio_host_mem_param) {
      i++;  // skip the next argument
    } else {
      _cudf_tmp_args.push_back(arg);
    }
  }

  args = _cudf_tmp_args;
}

}  // namespace cudf

// Install arg handler
#undef NVBENCH_MAIN_CUSTOM_ARGS_HANDLER
#define NVBENCH_MAIN_CUSTOM_ARGS_HANDLER(args) cudf::benchmark_arg_handler(args)

// Global fixture setup:
#undef NVBENCH_MAIN_INITIALIZE_CUSTOM_POST
#define NVBENCH_MAIN_INITIALIZE_CUSTOM_POST(argc, argv) \
  [[maybe_unused]] auto env_state = cudf::nvbench_base_fixture(argc, argv);

// this declares/defines the main() function using the definitions above
NVBENCH_MAIN
