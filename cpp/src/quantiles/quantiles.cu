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

#include <cudf/copying.hpp>
#include <cudf/detail/gather.cuh>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/detail/sorting.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/error.hpp>
#include <quantiles/quantiles_util.hpp>

#include <rmm/cuda_stream_view.hpp>

#include <memory>
#include <vector>

namespace cudf {
namespace detail {
template <typename SortMapIterator>
std::unique_ptr<table> quantiles(table_view const& input,
                                 SortMapIterator sortmap,
                                 std::vector<double> const& q,
                                 interpolation interp,
                                 rmm::cuda_stream_view stream,
                                 rmm::mr::device_memory_resource* mr)
{
  auto quantile_idx_lookup = [sortmap, interp, size = input.num_rows()] __device__(double q) {
    auto selector = [sortmap] __device__(auto idx) { return sortmap[idx]; };
    return detail::select_quantile<size_type>(selector, size, q, interp);
  };

  rmm::device_vector<double> q_device{q};

  auto quantile_idx_iter = thrust::make_transform_iterator(q_device.begin(), quantile_idx_lookup);

  return detail::gather(input, quantile_idx_iter, quantile_idx_iter + q.size(), false, stream, mr);
}

}  // namespace detail

std::unique_ptr<table> quantiles(table_view const& input,
                                 std::vector<double> const& q,
                                 interpolation interp,
                                 cudf::sorted is_input_sorted,
                                 std::vector<order> const& column_order,
                                 std::vector<null_order> const& null_precedence,
                                 rmm::mr::device_memory_resource* mr)
{
  CUDF_FUNC_RANGE();
  if (q.empty()) { return empty_like(input); }

  CUDF_EXPECTS(interp == interpolation::HIGHER || interp == interpolation::LOWER ||
                 interp == interpolation::NEAREST,
               "multi-column quantiles require a non-arithmetic interpolation strategy.");

  CUDF_EXPECTS(input.num_rows() > 0, "multi-column quantiles require at least one input row.");

  if (is_input_sorted == sorted::YES) {
    return detail::quantiles(
      input, thrust::make_counting_iterator<size_type>(0), q, interp, rmm::cuda_stream_default, mr);
  } else {
    auto sorted_idx = detail::sorted_order(input, column_order, null_precedence);
    return detail::quantiles(
      input, sorted_idx->view().data<size_type>(), q, interp, rmm::cuda_stream_default, mr);
  }
}

}  // namespace cudf
