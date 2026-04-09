/*
 * Copyright (c) 2026, NVIDIA CORPORATION.
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

#include <cudf/io/protobuf.hpp>
#include <cudf/utilities/export.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/resource_ref.hpp>

namespace CUDF_EXPORT cudf {
namespace io::protobuf::detail {

/**
 * @brief Check if an encoding is compatible with the given field and data type.
 */
bool is_encoding_compatible(nested_field_descriptor const& field, cudf::data_type const& type);

/**
 * @brief Validate the decode context (schema consistency, encoding compatibility, etc.).
 *
 * @throws cudf::logic_error if the context is invalid
 */
void validate_decode_options(decode_protobuf_options const& options);

/**
 * @brief Create a view into a single field's metadata from the decode options.
 */
protobuf_field_meta_view make_field_meta_view(decode_protobuf_options const& options,
                                              int schema_idx);

/**
 * @brief Internal implementation of decode_protobuf.
 */
std::unique_ptr<cudf::column> decode_protobuf(cudf::column_view const& binary_input,
                                              decode_protobuf_options const& options,
                                              rmm::cuda_stream_view stream,
                                              rmm::device_async_resource_ref mr);

}  // namespace io::protobuf::detail
}  // namespace CUDF_EXPORT cudf
