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

#include <optional>

namespace CUDF_EXPORT cudf {
//! IO interfaces
namespace io {

class kvikio_manager {
 public:
  ~kvikio_manager() = default;

  kvikio_manager(const kvikio_manager&)            = delete;
  kvikio_manager& operator=(const kvikio_manager&) = delete;
  kvikio_manager(kvikio_manager&&)                 = delete;
  kvikio_manager& operator=(kvikio_manager&&)      = delete;

  static kvikio_manager& instance();

  static void set_num_io_threads(unsigned int num_io_threads);

  static unsigned int get_num_io_threads();

 private:
  kvikio_manager();
  unsigned int _num_io_threads;
};

}  // namespace io
}  // namespace CUDF_EXPORT cudf
