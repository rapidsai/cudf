/*
 * Copyright (c) 2021, NVIDIA CORPORATION.
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

#include <cudf/groupby.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/span.hpp>

#include <rmm/cuda_stream_view.hpp>

#include <memory>
#include <utility>

namespace cudf {
namespace groupby {
namespace detail {

/**
 * TBA
 */
std::unique_ptr<column> group_shift_impl(
  column_view const& values,
  size_type offset,
  cudf::device_span<size_type const> group_offsets,
  cudf::device_span<size_type const> group_sizes,
  cudf::scalar const& fill_value,
  rmm::cuda_stream_view stream,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource());

/**
 * @copydoc TBA
 */
std::unique_ptr<column> group_shift(
  column_view const& values,
  size_type offset,
  scalar const& fill_value,
  device_span<size_type const> group_offsets,
  rmm::cuda_stream_view stream,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource());

}  // namespace detail
}  // namespace groupby
}  // namespace cudf
