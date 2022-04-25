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

#include <nvcomp.h>

namespace cudf::io::nvcomp {

struct batched_inputs {
  rmm::device_uvector<void const*> compressed_data_ptrs;
  rmm::device_uvector<size_t> compressed_data_sizes;
  rmm::device_uvector<void*> uncompressed_data_ptrs;
  rmm::device_uvector<size_t> uncompressed_data_sizes;
};

batched_inputs create_batched_inputs(device_span<device_decompress_input const> cudf_comp_in,
                                     rmm::cuda_stream_view stream);

__host__ void convert_status(device_span<nvcompStatus_t const> nvcomp_stats,
                             device_span<size_t const> actual_uncompressed_sizes,
                             device_span<decompress_status> cudf_stats,
                             rmm::cuda_stream_view stream);
}  // namespace cudf::io::nvcomp
