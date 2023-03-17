/*
 * Copyright (c) 2022-2023, NVIDIA CORPORATION.
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

#include "simple.cuh"

#include <cudf/reduction/detail/reduction_functions.hpp>

namespace cudf {
namespace reduction {
namespace detail {

std::unique_ptr<cudf::column> segmented_min(
  column_view const& col,
  device_span<size_type const> offsets,
  data_type const output_dtype,
  null_policy null_handling,
  std::optional<std::reference_wrapper<scalar const>> init,
  rmm::cuda_stream_view stream,
  rmm::mr::device_memory_resource* mr)
{
  CUDF_EXPECTS(col.type() == output_dtype,
               "segmented_min() operation requires matching output type");
  using reducer = simple::detail::same_column_type_dispatcher<op::min>;
  return cudf::type_dispatcher(
    col.type(), reducer{}, col, offsets, null_handling, init, stream, mr);
}
}  // namespace detail
}  // namespace reduction
}  // namespace cudf
