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

#include "io/utilities/parsing_utils.cuh"

#include <cudf/types.hpp>
#include <cudf/utilities/span.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/resource_ref.hpp>

#include <thrust/iterator/zip_iterator.h>
#include <thrust/tuple.h>

namespace cudf::io {
namespace detail {

/**
 * @brief Infers data type for a given JSON string input `data`.
 *
 * @throw cudf::logic_error if input size is 0
 * @throw cudf::logic_error if date time is not inferred as string
 * @throw cudf::logic_error if data type inference failed
 *
 * @param options View of inference options
 * @param data JSON string input
 * @param offset_length_begin The beginning of an offset-length tuple sequence
 * @param size Size of the string input
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @return The inferred data type
 */
cudf::data_type infer_data_type(
  cudf::io::json_inference_options_view const& options,
  device_span<char const> data,
  thrust::zip_iterator<thrust::tuple<const size_type*, const size_type*>> offset_length_begin,
  std::size_t const size,
  rmm::cuda_stream_view stream);
}  // namespace detail

namespace json::detail {

/**
 * @brief Parses the data from an iterator of string views, casting it to the given target data type
 *
 * @param data string input base pointer
 * @param offset_length_begin The beginning of an offset-length tuple sequence
 * @param col_size The total number of items of this column
 * @param col_type The column's target data type
 * @param null_mask A null mask that renders certain items from the input invalid
 * @param options Settings for controlling the processing behavior
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr The resource to be used for device memory allocation
 * @return The column that contains the parsed data
 */
std::unique_ptr<column> parse_data(
  const char* data,
  thrust::zip_iterator<thrust::tuple<const size_type*, const size_type*>> offset_length_begin,
  size_type col_size,
  data_type col_type,
  rmm::device_buffer&& null_mask,
  size_type null_count,
  cudf::io::parse_options_view const& options,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr);
}  // namespace json::detail
}  // namespace cudf::io
