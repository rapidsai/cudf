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

#include "update_validity.hpp"

#include <cudf/detail/null_mask.cuh>
#include <cudf/scalar/scalar.hpp>
#include <cudf/utilities/span.hpp>

#include <rmm/resource_ref.hpp>

namespace cudf {
namespace reduction {
namespace detail {

void segmented_update_validity(column& result,
                               column_view const& col,
                               device_span<size_type const> offsets,
                               null_policy null_handling,
                               std::optional<std::reference_wrapper<scalar const>> init,
                               rmm::cuda_stream_view stream,
                               rmm::device_async_resource_ref mr)
{
  auto [output_null_mask, output_null_count] = cudf::detail::segmented_null_mask_reduction(
    col.null_mask(),
    offsets.begin(),
    offsets.end() - 1,
    offsets.begin() + 1,
    null_handling,
    init.has_value() ? std::optional(init.value().get().is_valid(stream)) : std::nullopt,
    stream,
    mr);
  result.set_null_mask(std::move(output_null_mask), output_null_count);
}

}  // namespace detail
}  // namespace reduction
}  // namespace cudf
