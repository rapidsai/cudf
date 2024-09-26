/*
 * Copyright (c) 2024, NVIDIA CORPORATION.
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
#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_factories.hpp>
#include <cudf/detail/copy.hpp>
#include <cudf/detail/iterator.cuh>
#include <cudf/detail/null_mask.hpp>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/detail/utilities/cuda.cuh>
#include <cudf/detail/utilities/vector_factories.hpp>
#include <cudf/strings/detail/utilities.hpp>
#include <cudf/strings/find.hpp>
#include <cudf/strings/string_view.cuh>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/exec_policy.hpp>

#include <cub/cub.cuh>
#include <cuda/functional>
#include <thrust/binary_search.h>
#include <thrust/fill.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/logical.h>
#include <thrust/sequence.h>
#include <thrust/unique.h>

#include <algorithm>

namespace cudf {
namespace strings {
namespace detail {
namespace {

/**
 * @brief Threshold to decide on using string or warp parallel functions.
 *
 * If the average byte length of a string in a column exceeds this value then
 * a warp-parallel function is used.
 */
constexpr size_type AVG_CHAR_BYTES_THRESHOLD = 64;

CUDF_KERNEL void multi_contains_warp_parallel(column_device_view const d_strings,
                                              column_device_view const d_targets,
                                              u_char const* d_first_bytes,
                                              size_type const* d_indices,
                                              size_type const* d_offsets,
                                              size_type unique_count,
                                              bool* working_memory,
                                              cudf::device_span<bool*> d_results)
{
  auto const num_targets = d_targets.size();
  auto const idx         = cudf::detail::grid_1d::global_thread_id();
  auto const str_idx     = idx / cudf::detail::warp_size;
  if (str_idx >= d_strings.size()) { return; }
  if (d_strings.is_null(str_idx)) { return; }
  // get the string for this warp
  auto const d_str = d_strings.element<string_view>(str_idx);

  auto const lane_idx = idx % cudf::detail::warp_size;

  // size of shared_bools = num_targets * block_size
  // each thread uses num_targets bools
  extern __shared__ bool shared_bools[];
  auto const warp_idx = threadIdx.x / cudf::detail::warp_size;
  // bools for the current string
  auto bools = working_memory == nullptr
                 ? (shared_bools + (warp_idx * cudf::detail::warp_size * num_targets))
                 : (working_memory + (str_idx * cudf::detail::warp_size * num_targets));

  // initialize result: set true if target is empty, false otherwise
  for (auto target_idx = lane_idx; target_idx < num_targets;
       target_idx += cudf::detail::warp_size) {
    auto const d_target = d_targets.element<string_view>(target_idx);
    auto const begin    = bools + (target_idx * cudf::detail::warp_size);
    thrust::uninitialized_fill(
      thrust::seq, begin, begin + cudf::detail::warp_size, d_target.empty());
  }

  auto const last_ptr = d_first_bytes + unique_count;
  for (size_type str_byte_idx = lane_idx; str_byte_idx < d_str.size_bytes();
       str_byte_idx += cudf::detail::warp_size) {
    // search for byte in first_bytes array
    auto const chr      = static_cast<u_char>(*(d_str.data() + str_byte_idx));
    auto const byte_ptr = thrust::lower_bound(thrust::seq, d_first_bytes, last_ptr, chr);
    // if not found, continue to next byte
    if ((byte_ptr == last_ptr) || (*byte_ptr != chr)) { continue; }
    // compute index of matched byte
    auto const offset_idx = static_cast<size_type>(thrust::distance(d_first_bytes, byte_ptr));
    auto map_idx          = d_offsets[offset_idx];
    auto const last_idx = (offset_idx + 1) < unique_count ? d_offsets[offset_idx + 1] : num_targets;
    // check for targets that begin with chr
    while (map_idx < last_idx) {
      auto const target_idx = d_indices[map_idx++];
      auto const bool_idx   = (target_idx * cudf::detail::warp_size) + lane_idx;
      if (!bools[bool_idx]) {  // not found before
        auto const d_target = d_targets.element<string_view>(target_idx);
        if ((d_str.size_bytes() - str_byte_idx) >= d_target.size_bytes()) {
          // first char already checked, only need to check the [2nd, end) chars if has.
          bool found = true;
          for (auto i = 1; i < d_target.size_bytes() && found; i++) {
            if (*(d_str.data() + str_byte_idx + i) != *(d_target.data() + i)) { found = false; }
          }
          if (found) { bools[bool_idx] = true; }
        }
      }
    }
  }

  // wait all lanes are done in a warp
  __syncwarp();

  // reduce the bools for each target to store in the result
  for (auto target_idx = lane_idx; target_idx < num_targets;
       target_idx += cudf::detail::warp_size) {
    auto begin = bools + (target_idx * cudf::detail::warp_size);
    auto found =
      thrust::any_of(thrust::seq, begin, begin + cudf::detail::warp_size, thrust::identity<bool>{});
    d_results[target_idx][str_idx] = found;
  }
}

CUDF_KERNEL void multi_contains_row_parallel(column_device_view const d_strings,
                                             column_device_view const d_targets,
                                             u_char const* d_first_bytes,
                                             size_type const* d_indices,
                                             size_type const* d_offsets,
                                             size_type unique_count,
                                             cudf::device_span<bool*> d_results)
{
  auto const str_idx     = static_cast<size_type>(cudf::detail::grid_1d::global_thread_id());
  auto const num_targets = d_targets.size();
  if (str_idx >= d_strings.size()) { return; }
  if (d_strings.is_null(str_idx)) { return; }
  auto const d_str = d_strings.element<string_view>(str_idx);

  // initialize output; the result of searching empty target is true
  for (auto target_idx = 0; target_idx < num_targets; ++target_idx) {
    auto const d_target            = d_targets.element<string_view>(target_idx);
    d_results[target_idx][str_idx] = d_target.empty();
  }

  // process each byte of the current string
  auto const last_ptr = d_first_bytes + unique_count;
  for (auto str_byte_idx = 0; str_byte_idx < d_str.size_bytes(); ++str_byte_idx) {
    // search for byte in first_bytes array
    auto const chr      = static_cast<u_char>(*(d_str.data() + str_byte_idx));
    auto const byte_ptr = thrust::lower_bound(thrust::seq, d_first_bytes, last_ptr, chr);
    // if not found, continue to next byte
    if ((byte_ptr == last_ptr) || (*byte_ptr != chr)) { continue; }
    // compute index of matched byte
    auto const offset_idx = static_cast<size_type>(thrust::distance(d_first_bytes, byte_ptr));
    auto map_idx          = d_offsets[offset_idx];
    auto const last_idx = (offset_idx + 1) < unique_count ? d_offsets[offset_idx + 1] : num_targets;
    // check for targets that begin with chr
    while (map_idx < last_idx) {
      auto const target_idx = d_indices[map_idx++];
      if (!d_results[target_idx][str_idx]) {  // not found before
        auto const d_target = d_targets.element<string_view>(target_idx);
        if ((d_str.size_bytes() - str_byte_idx) >= d_target.size_bytes()) {
          // first char already checked, only need to check the [2nd, end) chars
          bool found = true;
          for (auto i = 1; i < d_target.size_bytes() && found; i++) {
            if (*(d_str.data() + str_byte_idx + i) != *(d_target.data() + i)) { found = false; }
          }
          if (found) { d_results[target_idx][str_idx] = true; }
        }
      }
    }
  }
}

}  // namespace

std::unique_ptr<table> contains_multiple(strings_column_view const& input,
                                         strings_column_view const& targets,
                                         rmm::cuda_stream_view stream,
                                         rmm::mr::device_memory_resource* mr)
{
  CUDF_EXPECTS(not targets.is_empty(), "Must specify at least one target string.");
  CUDF_EXPECTS(not targets.has_nulls(), "Target strings cannot be null");

  auto const num_targets = targets.size();

  auto const d_strings = column_device_view::create(input.parent(), stream);
  auto const d_targets = column_device_view::create(targets.parent(), stream);

  // copy the first byte of each target and sort them
  auto first_bytes = rmm::device_uvector<u_char>(targets.size(), stream);
  auto indices     = rmm::device_uvector<size_type>(targets.size(), stream);
  {
    auto tgt_itr = thrust::make_transform_iterator(
      d_targets->begin<string_view>(),
      cuda::proclaim_return_type<u_char>([] __device__(auto const& d_tgt) -> u_char {
        return d_tgt.empty() ? u_char{0} : static_cast<u_char>(d_tgt.data()[0]);
      }));
    auto count_itr = thrust::make_counting_iterator<size_type>(0);
    auto keys_out  = first_bytes.begin();
    auto vals_out  = indices.begin();
    auto cmp_op    = thrust::less();
    auto sv        = stream.value();

    std::size_t tmp_bytes = 0;
    cub::DeviceMergeSort::SortPairsCopy(
      nullptr, tmp_bytes, tgt_itr, count_itr, keys_out, vals_out, num_targets, cmp_op, sv);
    auto tmp_stg = rmm::device_buffer(tmp_bytes, stream);
    cub::DeviceMergeSort::SortPairsCopy(
      tmp_stg.data(), tmp_bytes, tgt_itr, count_itr, keys_out, vals_out, num_targets, cmp_op, sv);
  }

  // remove duplicates to speed up lower_bound
  auto offsets = rmm::device_uvector<size_type>(targets.size(), stream);
  thrust::sequence(rmm::exec_policy_nosync(stream), offsets.begin(), offsets.end());
  auto end = thrust::unique_by_key(
    rmm::exec_policy_nosync(stream), first_bytes.begin(), first_bytes.end(), offsets.begin());
  auto ucount = static_cast<size_type>(thrust::distance(first_bytes.begin(), end.first));

  // create output columns
  auto const results_iter = cudf::detail::make_counting_transform_iterator(0, [&](int i) {
    return make_numeric_column(data_type{type_id::BOOL8},
                               input.size(),
                               cudf::detail::copy_bitmask(input.parent(), stream, mr),
                               input.null_count(),
                               stream,
                               mr);
  });
  auto results_list =
    std::vector<std::unique_ptr<column>>(results_iter, results_iter + targets.size());
  auto device_results_list = [&] {
    auto host_results_pointer_iter =
      thrust::make_transform_iterator(results_list.begin(), [](auto const& results_column) {
        return results_column->mutable_view().template data<bool>();
      });
    auto host_results_pointers = std::vector<bool*>(
      host_results_pointer_iter, host_results_pointer_iter + results_list.size());
    return cudf::detail::make_device_uvector_async(host_results_pointers, stream, mr);
  }();

  constexpr cudf::thread_index_type block_size = 256;

  auto d_first_bytes = first_bytes.data();
  auto d_indices     = indices.data();
  auto d_offsets     = offsets.data();

  // Smaller strings perform better with a row per string
  bool const row_parallel = ((input.null_count() == input.size()) ||
                             ((input.chars_size(stream) / (input.size() - input.null_count())) <=
                              AVG_CHAR_BYTES_THRESHOLD));
  if (row_parallel) {
    cudf::detail::grid_1d grid{static_cast<cudf::thread_index_type>(input.size()), block_size};
    multi_contains_row_parallel<<<grid.num_blocks, grid.num_threads_per_block, 0, stream.value()>>>(
      *d_strings, *d_targets, d_first_bytes, d_indices, d_offsets, ucount, device_results_list);
  } else {
    cudf::detail::grid_1d grid{
      static_cast<cudf::thread_index_type>(input.size()) * cudf::detail::warp_size, block_size};
    auto shared_mem_size    = block_size * targets.size();
    size_type work_mem_size = 0;
    if (shared_mem_size > (16 * block_size)) {  // TODO: Need to find a good value for this
      shared_mem_size = 0;
      work_mem_size   = targets.size() * input.size() * cudf::detail::warp_size;
    }
    auto working_memory = rmm::device_uvector<bool>(work_mem_size, stream);
    multi_contains_warp_parallel<<<grid.num_blocks,
                                   grid.num_threads_per_block,
                                   shared_mem_size,
                                   stream.value()>>>(*d_strings,
                                                     *d_targets,
                                                     d_first_bytes,
                                                     d_indices,
                                                     d_offsets,
                                                     ucount,
                                                     working_memory.data(),
                                                     device_results_list);
  }

  return std::make_unique<table>(std::move(results_list));
}

}  // namespace detail

std::unique_ptr<table> contains_multiple(strings_column_view const& strings,
                                         strings_column_view const& targets,
                                         rmm::cuda_stream_view stream,
                                         rmm::mr::device_memory_resource* mr)
{
  CUDF_FUNC_RANGE();
  return detail::contains_multiple(strings, targets, stream, mr);
}

}  // namespace strings
}  // namespace cudf
