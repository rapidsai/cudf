/*
 * Copyright (c) 2019-2021, NVIDIA CORPORATION.
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

#include "scan.hpp"

#include <cudf/column/column_view.hpp>
#include <cudf/detail/aggregation/aggregation.hpp>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/reduction.hpp>
#include <cudf/utilities/error.hpp>

#include <rmm/cuda_stream_view.hpp>

namespace cudf {
namespace detail {

std::unique_ptr<column> scan(
  const column_view& input,
  std::unique_ptr<aggregation> const& agg,
  scan_type inclusive,
  null_policy null_handling,
  rmm::cuda_stream_view stream,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource())
{
  CUDF_EXPECTS(
    is_numeric(input.type()) || is_compound(input.type()) || is_fixed_point(input.type()),
    "Unexpected non-numeric or non-string type.");

  switch (agg->kind) {
    case aggregation::SUM: return scan_sum(input, inclusive, null_handling, stream, mr);
    case aggregation::MIN: return scan_min(input, inclusive, null_handling, stream, mr);
    case aggregation::MAX: return scan_max(input, inclusive, null_handling, stream, mr);
    case aggregation::PRODUCT: return scan_product(input, inclusive, null_handling, stream, mr);
    default: CUDF_FAIL("Unsupported aggregation operator for scan");
  }
}
}  // namespace detail

std::unique_ptr<column> scan(const column_view& input,
                             std::unique_ptr<aggregation> const& agg,
                             scan_type inclusive,
                             null_policy null_handling,
                             rmm::mr::device_memory_resource* mr)
{
  CUDF_FUNC_RANGE();
  return detail::scan(input, agg, inclusive, null_handling, rmm::cuda_stream_default, mr);
}

}  // namespace cudf
