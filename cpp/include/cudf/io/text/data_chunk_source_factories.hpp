/*
 * Copyright (c) 2021-2022, NVIDIA CORPORATION.
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

#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/io/text/data_chunk_source.hpp>
#include <cudf/scalar/scalar.hpp>

#include <rmm/device_buffer.hpp>

#include <thrust/host_vector.h>
#include <thrust/system/cuda/experimental/pinned_allocator.h>

#include <fstream>
#include <memory>
#include <sstream>
#include <string>
#include <unordered_map>

namespace cudf {
namespace io {
namespace detail {
class cufile_input_impl;
}
}  // namespace io
}  // namespace cudf

namespace cudf {
namespace io {
namespace text {

/**
 * @brief Creates a data source capable of producing device-buffered views of the given string.
 */
std::unique_ptr<data_chunk_source> make_source(std::string const& data);

/**
 * @brief Creates a data source capable of producing device-buffered views of the file
 */
std::unique_ptr<data_chunk_source> make_source_from_file(std::string const& filename);

/**
 * @brief Creates a data source capable of producing views of the given device string scalar
 */
std::unique_ptr<data_chunk_source> make_source(cudf::string_scalar& data);

}  // namespace text
}  // namespace io
}  // namespace cudf
