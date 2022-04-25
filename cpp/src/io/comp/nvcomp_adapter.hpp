/*
 * Copyright (c) 2022, NVIDIA CORPORATION.
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

#include "gpuinflate.h"

#include <cudf/utilities/span.hpp>

#include <rmm/cuda_stream_view.hpp>

namespace cudf::io::nvcomp {

enum class compression_type { SNAPPY };

void batched_decompress(compression_type type,
                        device_span<device_decompress_input const> comp_in,
                        device_span<decompress_status> comp_stat,
                        size_t max_uncomp_page_size,
                        rmm::cuda_stream_view stream);
}  // namespace cudf::io::nvcomp
