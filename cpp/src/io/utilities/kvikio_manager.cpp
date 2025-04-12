/*
 * Copyright (c) 2025, NVIDIA CORPORATION.
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

#include "getenv_or.hpp"

#include <cudf/io/kvikio_manager.hpp>
#include <cudf/logger.hpp>
#include <cudf/utilities/error.hpp>

#include <kvikio/defaults.hpp>

#include <cuda_runtime.h>

namespace cudf {
namespace io {

kvikio_manager::kvikio_manager()
{
  // Workaround for https://github.com/rapidsai/cudf/issues/14140, where cuFileDriverOpen errors
  // out if no CUDA calls have been made before it. This is a no-op if the CUDA context is already
  // initialized.
  cudaFree(nullptr);

  auto const compat_mode = kvikio::getenv_or("KVIKIO_COMPAT_MODE", kvikio::CompatMode::ON);
  kvikio::defaults::set_compat_mode(compat_mode);

  std::string const env_var_name{"KVIKIO_NTHREADS"};
  auto const env_val = std::getenv(env_var_name.c_str());
  if (env_val != nullptr) {
    // If the env var KVIKIO_NTHREADS is set, KvikIO will create the thread pool with
    // KVIKIO_NTHREADS when the singleton class kvikio::defaults is first instantiated via the above
    // kvikio::defaults::set_compat_mode() call. In this case, cuDF's default value is not derived
    // here so as to avoid the overhead of NVML initialization.
    CUDF_LOG_INFO("Environment variable %.*s read as %s",
                  static_cast<int>(env_var_name.length()),
                  env_var_name.data(),
                  env_val);
  } else {
    // If the env var KVIKIO_NTHREADS is not set, KvikIO will create the thread pool with its own
    // default value M. Here if cuDF's derived default value N is not equal to M, the existing
    // thread pool will be destroyed and a new one with N threads created.
    auto default_val = get_default_num_io_threads();
    std::stringstream ss;
    ss << default_val;
    CUDF_LOG_INFO("Environment variable %.*s is not set, using default value %s",
                  static_cast<int>(env_var_name.length()),
                  env_var_name.data(),
                  ss.str());
    set_num_io_threads(default_val);
  }
}

kvikio_manager& kvikio_manager::instance()
{
  static kvikio_manager kvikio_manager_instance;
  return kvikio_manager_instance;
}

void kvikio_manager::set_num_io_threads(unsigned int num_io_threads)
{
  if (num_io_threads != kvikio::defaults::thread_pool_nthreads()) {
    kvikio::defaults::set_thread_pool_nthreads(num_io_threads);
  }
}

unsigned int kvikio_manager::get_num_io_threads()
{
  return kvikio::defaults::thread_pool_nthreads();
}

unsigned int kvikio_manager::get_default_num_io_threads()
{
  // todo: provide a platform-dependent, default setting that can achieve decent performance for
  // single-process and multi-process benchmarks.
  return 4u;
}

}  // namespace io
}  // namespace cudf
