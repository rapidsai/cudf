/*
 * Copyright (c) 2019-20, NVIDIA CORPORATION.
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

#include "sort_impl.cuh"

#include <cudf/column/column.hpp>
#include <cudf/detail/sorting.hpp>
#include <cudf/sorting.hpp>
#include <cudf/table/table_view.hpp>

namespace cudf {
namespace detail {
std::unique_ptr<column> stable_sorted_order(table_view input,
                                            std::vector<order> const& column_order,
                                            std::vector<null_order> const& null_precedence,
                                            rmm::mr::device_memory_resource* mr,
                                            cudaStream_t stream)
{
  return sorted_order<true>(input, column_order, null_precedence, mr, stream);
}

}  // namespace detail

std::unique_ptr<column> stable_sorted_order(table_view input,
                                            std::vector<order> const& column_order,
                                            std::vector<null_order> const& null_precedence,
                                            rmm::mr::device_memory_resource* mr)
{
  return detail::stable_sorted_order(input, column_order, null_precedence, mr);
}

}  // namespace cudf
