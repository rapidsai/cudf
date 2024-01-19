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

#pragma once

#include "orc_gpu.hpp"

#include <io/utilities/hostdevice_vector.hpp>

#include <cudf/types.hpp>

#include <rmm/device_buffer.hpp>
#include <rmm/device_uvector.hpp>

namespace cudf::io::orc::detail {

/**
 * @brief Struct to store file-level data that remains constant for all chunks being read.
 */
struct file_intermediate_data {
  std::vector<std::vector<rmm::device_buffer>> lvl_stripe_data;
  std::vector<std::vector<rmm::device_uvector<uint32_t>>> null_count_prefix_sums;

  int64_t rows_to_skip;
  size_type rows_to_read;
  std::vector<metadata::stripe_source_mapping> selected_stripes;
};

}  // namespace cudf::io::orc::detail
