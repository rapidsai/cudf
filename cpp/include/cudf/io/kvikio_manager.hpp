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

#pragma once

#include <cudf/utilities/export.hpp>

namespace CUDF_EXPORT cudf {
//! KvikIO manager
namespace io {

namespace detail {
/**
 * @brief Utility class for data source and sink.
 *
 * Use this class as a data member to create a CUDA context if it has not existed, and create a
 * instance of kvikio_manager.
 */
class kvikio_initializer {
 public:
  kvikio_initializer();
};

}  // namespace detail

/**
 * @addtogroup io_kvikio_manager
 * @{
 * @file
 */

/**
 * @brief Singleton class to control the KvikIO library.
 */
class kvikio_manager {
 public:
  ~kvikio_manager() = default;

  kvikio_manager(const kvikio_manager&)            = delete;
  kvikio_manager& operator=(const kvikio_manager&) = delete;
  kvikio_manager(kvikio_manager&&)                 = delete;
  kvikio_manager& operator=(kvikio_manager&&)      = delete;

  /**
   * @brief Get the singleton instance of KvikIO manager.
   *
   * @return KvikIO manager instance.
   */
  static kvikio_manager& instance();

  /**
   * @brief Set the number of IO threads used by the KvikIO manager.
   *
   * If the new value differs from the current one, the following happens in sequence:
   *
   * - The calling thread is blocked until all pending I/O tasks complete.
   * - The old thread pool is destroyed.
   * - A new pool is created.
   *
   * Otherwise, the existing thread pool will be used for subsequent I/O operations.
   *
   * @param num_io_threads The number of IO threads to be used.
   *
   * @note This function instantiates the kvikio_manager if it does not exist.
   */
  static void set_num_io_threads(unsigned int num_io_threads);

  /**
   * @brief Get the number of IO threads used by the KvikIO manager.
   *
   * @return The number of IO threads used by the KvikIO manager.
   *
   * @note This function instantiates the kvikio_manager if it does not exist.
   */
  [[nodiscard]] static unsigned int num_io_threads();

  /**
   * @brief Get the default number of IO threads derived by cuDF.
   *
   * @return The default number of IO threads derived by cuDF.
   *
   * @note This function does not instantiate the kvikio_manager if it does not exist.
   */
  [[nodiscard]] static unsigned int default_num_io_threads();

 private:
  /**
   * @brief Constructor of kvikio_manager.
   */
  kvikio_manager();

  unsigned int _num_io_threads;
};

/** @} */  // end of group
}  // namespace io
}  // namespace CUDF_EXPORT cudf
