/*
 * Copyright (c) 2020, NVIDIA CORPORATION.
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

// #include <cudf/column/column_factories.hpp>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/detail/round.hpp>
#include <cudf/round.hpp>
#include <cudf/scalar/scalar.hpp>
// #include <cudf/scalar/scalar_factories.hpp>
// #include <cudf/utilities/error.hpp>
// #include <cudf/utilities/traits.hpp>
// #include <cudf/fixed_point/fixed_point.hpp>
#include <cudf/types.hpp>

namespace cudf {

namespace detail {

// TODO docs
std::unique_ptr<column> round(column_view const& col,
                              int32_t scale,
                              cudf::round_option round,
                              rmm::mr::device_memory_resource* mr,
                              cudaStream_t stream)
{
  return nullptr;  // TODO
}

}  // namespace detail

std::unique_ptr<column> round(column_view const& col,
                              int32_t scale,
                              round_option round,
                              rmm::mr::device_memory_resource* mr)
{
  CUDF_FUNC_RANGE();
  return cudf::detail::round(col, scale, round, mr);
}

}  // namespace cudf
