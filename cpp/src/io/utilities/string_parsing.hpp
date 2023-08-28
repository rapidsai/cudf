/*
 * Copyright (c) 2023, NVIDIA CORPORATION.
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

#include <io/utilities/parsing_utils.cuh>

#include <cudf/types.hpp>
#include <cudf/utilities/span.hpp>

#include <rmm/cuda_stream_view.hpp>

#include <thrust/iterator/zip_iterator.h>
#include <thrust/tuple.h>

namespace cudf::io::detail {

cudf::data_type infer_data_type(
  cudf::io::json_inference_options_view const& options,
  device_span<char const> data,
  thrust::zip_iterator<thrust::tuple<const size_type*, const size_type*>> offset_length_begin,
  std::size_t const size,
  rmm::cuda_stream_view stream);
}
