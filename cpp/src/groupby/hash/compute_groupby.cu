/*
 * Copyright (c) 2024-2025, NVIDIA CORPORATION.
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

#include "compute_aggregations.hpp"
#include "compute_groupby.hpp"
#include "helpers.cuh"
#include "sparse_to_dense_results.hpp"

#include <cudf/detail/aggregation/aggregation.cuh>
#include <cudf/detail/aggregation/result_cache.hpp>
#include <cudf/detail/cuco_helpers.hpp>
#include <cudf/detail/gather.hpp>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/groupby.hpp>
#include <cudf/null_mask.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/span.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/mr/device/device_memory_resource.hpp>

#include <cuco/static_set.cuh>

#include <iterator>
#include <memory>

namespace cudf::groupby::detail::hash {

template <typename Equal, typename Hash>
std::unique_ptr<table> compute_groupby(table_view const& keys,
                                       host_span<aggregation_request const> requests,
                                       bool skip_rows_with_nulls,
                                       Equal const& d_row_equal,
                                       Hash const& d_row_hash,
                                       cudf::detail::result_cache* cache,
                                       rmm::cuda_stream_view stream,
                                       rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();

  // convert to int64_t to avoid potential overflow with large `keys`
  auto const num_keys = static_cast<int64_t>(keys.num_rows());

  // Cache of sparse results where the location of aggregate value in each
  // column is indexed by the hash set
  cudf::detail::result_cache sparse_results(requests.size());

  auto const [input_index_to_key_index, gather_map] = [&] {
    auto set = cuco::static_set{
      cuco::extent<int64_t>{num_keys},
      cudf::detail::CUCO_DESIRED_LOAD_FACTOR,  // 50% load factor
      cuco::empty_key{cudf::detail::CUDF_SIZE_TYPE_SENTINEL},
      d_row_equal,
      probing_scheme_t{d_row_hash},
      cuco::thread_scope_device,
      cuco::storage<GROUPBY_BUCKET_SIZE>{},
      cudf::detail::cuco_allocator<char>{rmm::mr::polymorphic_allocator<char>{}, stream},
      stream.value()};

    rmm::device_uvector<size_type> key_indices(num_keys, stream);
    auto set_ref = set.ref(cuco::op::insert_and_find);
    thrust::transform(rmm::exec_policy_nosync(stream),
                      thrust::make_counting_iterator(0),
                      thrust::make_counting_iterator(keys.num_rows()),
                      key_indices.begin(),
                      [set_ref] __device__(size_type const idx) mutable {
                        auto const [inserted_idx_ptr, _] = set_ref.insert_and_find(idx);
                        return *inserted_idx_ptr;
                      });
    rmm::device_uvector<cudf::size_type> gather_map(num_keys, stream);
    auto const keys_end = set.retrieve_all(gather_map.begin(), stream.value());
    gather_map.resize(std::distance(gather_map.begin(), keys_end), stream);
    return std::pair{std::move(key_indices), std::move(gather_map)};
  }();

  // {
  //   auto h_map = cudf::detail::make_std_vector(input_index_to_key_index, stream);
  //   printf("\n\n\nmap: \n");
  //   for (auto& idx : h_map) {
  //     printf("%d, ", idx);
  //   }
  //   printf("\n\n\n");
  // }

  auto set = cuco::static_set{
    cuco::extent<int64_t>{num_keys},
    cudf::detail::CUCO_DESIRED_LOAD_FACTOR,  // 50% load factor
    cuco::empty_key{cudf::detail::CUDF_SIZE_TYPE_SENTINEL},
    key_indices_comparator_t{input_index_to_key_index.begin()},
    simplified_probing_scheme_t{input_index_to_key_index.begin()},
    cuco::thread_scope_device,
    cuco::storage<GROUPBY_BUCKET_SIZE>{},
    cudf::detail::cuco_allocator<char>{rmm::mr::polymorphic_allocator<char>{}, stream},
    stream.value()};

  auto row_bitmask =
    skip_rows_with_nulls
      ? cudf::bitmask_and(keys, stream, cudf::get_current_device_resource_ref()).first
      : rmm::device_buffer{};

  // Compute all single pass aggs first
  compute_aggregations(num_keys,
                       skip_rows_with_nulls,
                       static_cast<bitmask_type*>(row_bitmask.data()),
                       set,
                       gather_map,
                       requests,
                       &sparse_results,
                       stream);

  // {
  //   auto h_map = cudf::detail::make_std_vector(gather_map, stream);
  //   printf("\n\n\ngather_map: \n");
  //   for (auto& idx : h_map) {
  //     printf("%d, ", idx);
  //   }
  //   printf("\n\n\n");
  // }

  // Compact all results from sparse_results and insert into cache
  sparse_to_dense_results(requests,
                          &sparse_results,
                          cache,
                          gather_map,
                          set.ref(cuco::find),
                          static_cast<bitmask_type*>(row_bitmask.data()),
                          stream,
                          mr);

  return cudf::detail::gather(keys,
                              gather_map,
                              out_of_bounds_policy::DONT_CHECK,
                              cudf::detail::negative_index_policy::NOT_ALLOWED,
                              stream,
                              mr);
}

template std::unique_ptr<table> compute_groupby<row_comparator_t, row_hash_t>(
  table_view const& keys,
  host_span<aggregation_request const> requests,
  bool skip_rows_with_nulls,
  row_comparator_t const& d_row_equal,
  row_hash_t const& d_row_hash,
  cudf::detail::result_cache* cache,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr);

template std::unique_ptr<table> compute_groupby<nullable_row_comparator_t, row_hash_t>(
  table_view const& keys,
  host_span<aggregation_request const> requests,
  bool skip_rows_with_nulls,
  nullable_row_comparator_t const& d_row_equal,
  row_hash_t const& d_row_hash,
  cudf::detail::result_cache* cache,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr);
}  // namespace cudf::groupby::detail::hash
