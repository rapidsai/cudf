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

#include <cudf/column/column.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_buffer.hpp>
#include <rmm/resource_ref.hpp>

#include <nanoarrow/nanoarrow.hpp>

namespace cudf {
namespace detail {

std::unique_ptr<column> string_column_from_arrow_host(ArrowSchemaView* schema,
                                                      ArrowArray const* input,
                                                      std::unique_ptr<rmm::device_buffer>&& mask,
                                                      rmm::cuda_stream_view stream,
                                                      rmm::device_async_resource_ref mr);

std::unique_ptr<column> get_column_from_host_copy(ArrowSchemaView* schema,
                                                  ArrowArray const* input,
                                                  data_type type,
                                                  bool skip_mask,
                                                  rmm::cuda_stream_view stream,
                                                  rmm::device_async_resource_ref mr);

}  // namespace detail
}  // namespace cudf
