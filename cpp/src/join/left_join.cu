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

#include <cudf/detail/cuco_helpers.hpp>
#include <cudf/detail/join/left_join.cuh>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_uvector.hpp>

#include <cuco/bucket_storage.cuh>
#include <cuco/extent.cuh>

#include <algorithm>

namespace cudf {
namespace detail {
namespace {
auto compute_bucket_storage_size(cudf::table_view tbl, double load_factor)
{
  return std::max({static_cast<cudf::size_type>(
                     cuco::make_valid_extent<left_join::primitive_probing_scheme,
                                             left_join::storage_type,
                                             cudf::size_type>(tbl.num_rows(), load_factor)),
                   static_cast<cudf::size_type>(
                     cuco::make_valid_extent<left_join::nested_probing_scheme,
                                             left_join::storage_type,
                                             cudf::size_type>(tbl.num_rows(), load_factor)),
                   static_cast<cudf::size_type>(
                     cuco::make_valid_extent<left_join::simple_probing_scheme,
                                             left_join::storage_type,
                                             cudf::size_type>(tbl.num_rows(), load_factor))});
}
}  // namespace

left_join::left_join(cudf::table_view const& build,
                     null_equality compare_nulls,
                     rmm::cuda_stream_view stream)
  // If we cannot know beforehand about null existence then let's assume that there are nulls.
  : left_join(build, compare_nulls, cudf::detail::CUCO_DESIRED_LOAD_FACTOR, stream)
{
}

left_join::left_join(cudf::table_view const& build,
                     null_equality compare_nulls,
                     double load_factor,
                     rmm::cuda_stream_view stream)
  : _has_nested_columns{cudf::has_nested_columns(build)},
    _nulls_equal{compare_nulls},
    _build{build},
    _preprocessed_build{
      cudf::experimental::row::equality::preprocessed_table::create(_build, stream)},
    _bucket_storage{cuco::extent<cudf::size_type>{compute_bucket_storage_size(build, load_factor)},
                    cuco_allocator<char>{rmm::mr::polymorphic_allocator<char>{}, stream.value()}}
{
}

}  // namespace detail
}  // namespace cudf
