/*
 * Copyright (c) 2020-2024, NVIDIA CORPORATION.
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

#include "join/join_common_utils.hpp"

#include <cudf/detail/gather.hpp>
#include <cudf/detail/iterator.cuh>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/detail/search.hpp>
#include <cudf/dictionary/detail/update_keys.hpp>
#include <cudf/join.hpp>
#include <cudf/table/table.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

#include <thrust/copy.h>
#include <thrust/distance.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/sequence.h>
#include <thrust/transform.h>

#include <memory>

namespace cudf {
namespace detail {

std::unique_ptr<rmm::device_uvector<cudf::size_type>> left_semi_anti_join(
  join_kind const kind,
  cudf::table_view const& left_keys,
  cudf::table_view const& right_keys,
  null_equality compare_nulls,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr)
{
  CUDF_EXPECTS(0 != left_keys.num_columns(), "Left table is empty");
  CUDF_EXPECTS(0 != right_keys.num_columns(), "Right table is empty");

  if (is_trivial_join(left_keys, right_keys, kind)) {
    return std::make_unique<rmm::device_uvector<cudf::size_type>>(0, stream, mr);
  }
  if ((join_kind::LEFT_ANTI_JOIN == kind) && (0 == right_keys.num_rows())) {
    auto result =
      std::make_unique<rmm::device_uvector<cudf::size_type>>(left_keys.num_rows(), stream, mr);
    thrust::sequence(rmm::exec_policy(stream), result->begin(), result->end());
    return result;
  }

  // Materialize a `flagged` boolean array to generate a gather map.
  // Previously, the gather map was generated directly without this array but by calling to
  // `map.contains` inside the `thrust::copy_if` kernel. However, that led to increasing register
  // usage and reducing performance, as reported here: https://github.com/rapidsai/cudf/pull/10511.
  auto const flagged = cudf::detail::contains(right_keys,
                                              left_keys,
                                              compare_nulls,
                                              nan_equality::ALL_EQUAL,
                                              stream,
                                              cudf::get_current_device_resource_ref());

  auto const left_num_rows = left_keys.num_rows();
  auto gather_map =
    std::make_unique<rmm::device_uvector<cudf::size_type>>(left_num_rows, stream, mr);

  // gather_map_end will be the end of valid data in gather_map
  auto gather_map_end =
    thrust::copy_if(rmm::exec_policy(stream),
                    thrust::counting_iterator<size_type>(0),
                    thrust::counting_iterator<size_type>(left_num_rows),
                    gather_map->begin(),
                    [kind, d_flagged = flagged.begin()] __device__(size_type const idx) {
                      return *(d_flagged + idx) == (kind == join_kind::LEFT_SEMI_JOIN);
                    });

  gather_map->resize(thrust::distance(gather_map->begin(), gather_map_end), stream);
  return gather_map;
}

}  // namespace detail

std::unique_ptr<rmm::device_uvector<cudf::size_type>> left_semi_join(
  cudf::table_view const& left,
  cudf::table_view const& right,
  null_equality compare_nulls,
  rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  return detail::left_semi_anti_join(
    detail::join_kind::LEFT_SEMI_JOIN, left, right, compare_nulls, cudf::get_default_stream(), mr);
}

std::unique_ptr<rmm::device_uvector<cudf::size_type>> left_anti_join(
  cudf::table_view const& left,
  cudf::table_view const& right,
  null_equality compare_nulls,
  rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  return detail::left_semi_anti_join(
    detail::join_kind::LEFT_ANTI_JOIN, left, right, compare_nulls, cudf::get_default_stream(), mr);
}

}  // namespace cudf
