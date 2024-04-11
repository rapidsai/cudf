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
#define NVBENCH_ENVIRONMENT cudf::nvbench_base_fixture

#include <nvbench/main.cuh>

#include <vector>

// strip off the rmm_mode and cuio_host_mem parameters before passing the
// remaining arguments to nvbench::option_parser
#undef NVBENCH_MAIN_PARSE
#define NVBENCH_MAIN_PARSE(argc, argv)                     \
  nvbench::option_parser parser;                           \
  std::vector<std::string> m_args;                         \
  for (int i = 0; i < argc; ++i) {                         \
    std::string arg = argv[i];                             \
    if (arg == cudf::detail::rmm_mode_param) {             \
      i += 2;                                              \
    } else if (arg == cudf::detail::cuio_host_mem_param) { \
      i += 2;                                              \
    } else {                                               \
      m_args.push_back(arg);                               \
    }                                                      \
  }                                                        \
  parser.parse(m_args)

// this declares/defines the main() function using the definitions above
NVBENCH_MAIN
