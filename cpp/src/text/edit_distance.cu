/*
 * Copyright (c) 2020-2025, NVIDIA CORPORATION.
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

#include <cudf/column/column.hpp>
#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_factories.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/detail/sequence.hpp>
#include <cudf/detail/sizes_to_offsets_iterator.cuh>
#include <cudf/detail/utilities/cuda.cuh>
#include <cudf/detail/utilities/grid_1d.cuh>
#include <cudf/detail/utilities/integer_utils.hpp>
#include <cudf/strings/string_view.cuh>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <nvtext/edit_distance.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

#include <cooperative_groups.h>
#include <cuda/std/functional>
#include <thrust/execution_policy.h>
#include <thrust/fill.h>
#include <thrust/for_each.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/sequence.h>
#include <thrust/transform.h>

namespace nvtext {
namespace detail {
namespace {

/**
 * @brief Compute the Levenshtein distance for each string pair
 *
 * Documentation here: https://www.cuelogic.com/blog/the-levenshtein-algorithm
 * And here: https://en.wikipedia.org/wiki/Levenshtein_distance
 *
 * @param d_str First string
 * @param d_tgt Second string
 * @param buffer Working buffer for intermediate calculations
 * @return The edit distance value
 */
__device__ cudf::size_type compute_distance(cudf::string_view const& d_str,
                                            cudf::string_view const& d_tgt,
                                            cudf::size_type* buffer)
{
  auto const str_length = d_str.length();
  auto const tgt_length = d_tgt.length();
  if (str_length == 0) return tgt_length;
  if (tgt_length == 0) return str_length;

  auto begin   = str_length < tgt_length ? d_str.begin() : d_tgt.begin();
  auto itr     = str_length < tgt_length ? d_tgt.begin() : d_str.begin();
  auto const n = cuda::std::min(str_length, tgt_length);
  auto const m = cuda::std::max(str_length, tgt_length);
  // setup compute buffer pointers
  auto v0 = buffer;
  auto v1 = v0 + n + 1;
  // initialize v0
  thrust::sequence(thrust::seq, v0, v1);

  for (int i = 0; i < m; ++i, ++itr) {
    auto itr_tgt = begin;
    v1[0]        = i + 1;
    for (int j = 0; j < n; ++j, ++itr_tgt) {
      auto sub_cost = v0[j] + (*itr != *itr_tgt);
      auto del_cost = v0[j + 1] + 1;
      auto ins_cost = v1[j] + 1;
      v1[j + 1]     = cuda::std::min(cuda::std::min(sub_cost, del_cost), ins_cost);
    }
    cuda::std::swap(v0, v1);
  }
  return v0[n];
}

struct calculate_compute_buffer_fn {
  cudf::column_device_view d_strings;
  cudf::column_device_view d_targets;
  int32_t pad;
  int32_t count;

  __device__ std::ptrdiff_t operator()(cudf::size_type idx) const
  {
    if (d_strings.is_null(idx)) { return 0; }
    if (d_targets.size() > 1 && d_targets.is_null(idx)) { return 0; }
    auto d_str = d_strings.element<cudf::string_view>(idx);
    auto d_tgt = d_targets.size() == 1 ? d_targets.element<cudf::string_view>(0)
                                       : d_targets.element<cudf::string_view>(idx);
    return (cuda::std::min(d_str.length(), d_tgt.length()) + static_cast<int64_t>(pad)) *
           static_cast<int64_t>(count);
  }
};

/**
 * @brief Processes the 2 given strings by computing the matrix values
 * comparing each character of both strings
 *
 * The full matrix is computed from top-left to bottom-right with the edit-distance
 * as the final value. Only 3 diagonals are needed in memory at any iteration.
 *
 *     -  S  E  C  O  N  D
 *  -  0  1  2  3  4  5  6
 *  F  1  =  x  y  z  ↘  .
 *  I  2  x  y  z  ↘  .  .
 *  R  3  y  z  ↘  .  .  .
 *  S  4  z  ↘  .  .  .  .
 *  T  5  →  .  .  .  .  X
 *
 * Computing a diagonal only requires the previous 2 and each value
 * can be computed independently/parallel a tile at a time.
 *
 * The diagonal approach inspired by:
 * https://ashvardanian.com/posts/stringwars-on-gpus/#dynamic-programming-and-levenshtein-evaluation-order
 */
template <int32_t tile_size = cudf::detail::warp_size, typename Iterator>
__device__ void compute_distance(Iterator ss1,
                                 cudf::size_type length1,
                                 Iterator ss2,
                                 cudf::size_type length2,
                                 cudf::size_type* d_buffer,
                                 cudf::size_type* d_result)
{
  namespace cg        = cooperative_groups;
  auto const block    = cg::this_thread_block();
  auto const tile     = cg::tiled_partition<tile_size>(block);
  auto const lane_idx = tile.thread_rank();

  // shortcut if one of the strings is empty
  // (null rows are mapped here as well)
  if (length1 == 0 || length2 == 0) {
    if (lane_idx == 0) { *d_result = length1 == 0 ? length2 : length1; }
    return;
  }

  // setup the 3 working vectors for this string
  auto v0 = d_buffer;
  auto v1 = v0 + length1 + 2;
  auto v2 = v1 + length1 + 2;
  if (lane_idx == 0) {
    v0[0] = 0;  // first diagonal
    v1[0] = 1;  // second diagonal
    v1[1] = 1;
  }

  // for navigating the diagonal of the matrix of characters for the 2 strings
  auto next_itr = [](Iterator sitr, cudf::size_type length, Iterator itr, cudf::size_type offset) {
    if constexpr (cuda::std::is_pointer_v<Iterator>) {
      itr = sitr - offset;  // ASCII iterator
    } else {
      auto const pos = sitr.position() - offset;  // minimizes character counting
      itr += (pos >= 0) && (pos < length) ? (pos - itr.position()) : 0;
    }
    return itr;
  };

  // top-left of the matrix
  for (auto idx = 0; idx < length1; ++idx, ++ss1) {
    auto const n = idx + 2;
    auto const a = n > length1;

    auto jdx = static_cast<cudf::size_type>(lane_idx);
    auto it1 = ss1;
    auto it2 = ss2;

    auto tile_count = cudf::util::div_rounding_up_unsafe(n, tile_size);
    while (tile_count--) {
      auto const offset = (jdx - 1);
      // locate the 2 characters to compare
      it1 = next_itr(ss1, length1, it1, offset);
      it2 = next_itr(ss2, length2, it2, -offset);
      if (jdx == 0) {
        if (!a) { v2[0] = n; }
      } else if (jdx < n) {
        auto const sc = v0[jdx - 1] + (*it1 != *it2);
        auto const dc = v1[jdx - 1] + 1;
        auto const ic = v1[jdx] + 1;
        v2[jdx - a]   = cuda::std::min(cuda::std::min(sc, dc), ic);
      } else if (jdx == n) {
        v2[n - a] = n;
      }
      tile.sync();
      jdx += tile_size;
    }
    cuda::std::swap(v0, v1);
    cuda::std::swap(v1, v2);
  }

  --ss1;  // reset
  ++ss2;  // iterators

  // bottom-right of the matrix
  for (auto idx = 1; idx < length2; ++idx, ++ss2) {
    bool const fl = (length2 - idx) > length1;
    auto const n  = (fl ? length1 : (length2 - idx)) + 1;

    auto jdx = static_cast<cudf::size_type>(lane_idx);
    auto it1 = ss1;
    auto it2 = ss2;

    auto tile_count = cudf::util::div_rounding_up_unsafe(n, tile_size);
    while (tile_count--) {
      auto const offset = (jdx - 1);
      // locate the 2 characters to compare
      it1 = next_itr(ss1, length1, it1, offset);
      it2 = next_itr(ss2, length2, it2, -offset);
      if (jdx > 0 && jdx < n) {
        auto const sc = v0[jdx] + (*it1 != *it2);
        auto const dc = v1[jdx - 1] + 1;
        auto const ic = v1[jdx] + 1;
        v2[jdx - 1]   = cuda::std::min(cuda::std::min(sc, dc), ic);
      } else if (jdx == n && fl) {
        v2[n - 1] = n + idx;
      }
      tile.sync();
      jdx += tile_size;
    }
    cuda::std::swap(v0, v1);
    cuda::std::swap(v1, v2);
  }

  if (lane_idx == 0) { *d_result = v1[0]; }
}

template <int32_t tile_size = cudf::detail::warp_size>
CUDF_KERNEL void levenshtein_kernel(cudf::column_device_view d_strings,
                                    cudf::column_device_view d_targets,
                                    cudf::size_type* d_work_buffer,
                                    std::ptrdiff_t const* d_offsets,
                                    cudf::size_type* d_results)
{
  auto const tid     = cudf::detail::grid_1d::global_thread_id();
  auto const str_idx = tid / tile_size;
  if (str_idx >= d_strings.size()) { return; }
  auto d_str1 = d_strings.is_null(str_idx) ? cudf::string_view{}
                                           : d_strings.element<cudf::string_view>(str_idx);
  auto d_str2 = [&] {  // d_targets is also allowed to have only one valid entry
    if (d_targets.size() > 1 && d_targets.is_null(str_idx)) { return cudf::string_view{}; }
    return d_targets.size() == 1 ? d_targets.element<cudf::string_view>(0)
                                 : d_targets.element<cudf::string_view>(str_idx);
  }();

  if (d_str1.length() > d_str2.length()) { cuda::std::swap(d_str1, d_str2); }
  auto const length1 = d_str1.length();
  auto const length2 = d_str2.length();

  auto d_buffer = d_work_buffer + d_offsets[str_idx];
  auto d_result = d_results + str_idx;

  if (length1 == d_str1.size_bytes() && length2 == d_str2.size_bytes()) {
    // ASCII path
    compute_distance(d_str1.data(), length1, d_str2.data(), length2, d_buffer, d_result);
  } else {
    // use UTF8 iterator builtin to cudf::string_view
    compute_distance(d_str1.begin(), length1, d_str2.begin(), length2, d_buffer, d_result);
  }
}

}  // namespace

std::unique_ptr<cudf::column> edit_distance(cudf::strings_column_view const& input,
                                            cudf::strings_column_view const& targets,
                                            rmm::cuda_stream_view stream,
                                            rmm::device_async_resource_ref mr)
{
  auto const output_type = cudf::data_type{cudf::type_to_id<cudf::size_type>()};
  if (input.is_empty()) { return cudf::make_empty_column(output_type); }
  if (targets.size() != 1) {
    CUDF_EXPECTS(input.size() == targets.size(),
                 "targets.size() must equal input.size()",
                 std::invalid_argument);
  } else {
    CUDF_EXPECTS(!targets.has_nulls(), "single target must not be null", std::invalid_argument);
  }

  // create device columns from the input columns
  auto d_strings = cudf::column_device_view::create(input.parent(), stream);
  auto d_targets = cudf::column_device_view::create(targets.parent(), stream);

  // calculate the size of the compute-buffer
  rmm::device_uvector<std::ptrdiff_t> offsets(input.size() + 1, stream);
  thrust::transform(rmm::exec_policy_nosync(stream),
                    thrust::counting_iterator<cudf::size_type>(0),
                    thrust::counting_iterator<cudf::size_type>(input.size()),
                    offsets.begin(),
                    calculate_compute_buffer_fn{*d_strings, *d_targets, 2, 3});

  // get the total size of the temporary compute buffer
  // and convert sizes to offsets in-place
  auto const compute_size =
    cudf::detail::sizes_to_offsets(offsets.begin(), offsets.end(), offsets.begin(), 0, stream);
  rmm::device_uvector<cudf::size_type> compute_buffer(compute_size, stream);
  auto d_buffer = compute_buffer.data();

  auto results = cudf::make_fixed_width_column(
    output_type, input.size(), rmm::device_buffer{0, stream, mr}, 0, stream, mr);
  auto d_results = results->mutable_view().data<cudf::size_type>();

  constexpr auto block_size = 256L;
  constexpr auto tile_size  = static_cast<cudf::thread_index_type>(cudf::detail::warp_size);
  cudf::detail::grid_1d grid{input.size() * tile_size, block_size};
  levenshtein_kernel<tile_size><<<grid.num_blocks, grid.num_threads_per_block, 0, stream.value()>>>(
    *d_strings, *d_targets, d_buffer, offsets.data(), d_results);

  return results;
}

namespace {
struct edit_distance_matrix_levenshtein_algorithm {
  cudf::column_device_view d_strings;  // computing these against itself
  cudf::size_type* d_buffer;           // compute buffer for each string
  std::ptrdiff_t const* d_offsets;     // locate sub-buffer for each string
  cudf::size_type* d_results;          // edit distance values

  __device__ void operator()(cudf::size_type idx) const
  {
    auto const strings_count = d_strings.size();
    auto const row           = idx / strings_count;
    auto const col           = idx % strings_count;
    if (row > col) return;  // bottom half is computed with the top half of matrix
    cudf::string_view d_str1 =
      d_strings.is_null(row) ? cudf::string_view{} : d_strings.element<cudf::string_view>(row);
    cudf::string_view d_str2 =
      d_strings.is_null(col) ? cudf::string_view{} : d_strings.element<cudf::string_view>(col);
    auto work_buffer    = d_buffer + d_offsets[idx - ((row + 1L) * (row + 2L)) / 2L];
    auto const distance = (row == col) ? 0 : compute_distance(d_str1, d_str2, work_buffer);
    d_results[idx]      = distance;                   // top half of matrix
    d_results[col * strings_count + row] = distance;  // bottom half of matrix
  }
};

struct calculate_matrix_compute_buffer_fn {
  cudf::column_device_view d_strings;
  std::ptrdiff_t* d_sizes;

  __device__ void operator()(cudf::size_type idx) const
  {
    auto const row = idx / d_strings.size();
    auto const col = idx % d_strings.size();
    if (row >= col) { return; }  // compute only the top half
    cudf::string_view const d_str1 =
      d_strings.is_null(row) ? cudf::string_view{} : d_strings.element<cudf::string_view>(row);
    cudf::string_view const d_str2 =
      d_strings.is_null(col) ? cudf::string_view{} : d_strings.element<cudf::string_view>(col);
    if (d_str1.empty() || d_str2.empty()) { return; }
    // the temp size needed is 2 integers per character of the shorter string
    d_sizes[idx - ((row + 1L) * (row + 2L)) / 2L] =
      (cuda::std::min(d_str1.length(), d_str2.length()) + 1L) * 2L;
  }
};

}  // namespace

/**
 * @copydoc nvtext::edit_distance_matrix
 */
std::unique_ptr<cudf::column> edit_distance_matrix(cudf::strings_column_view const& input,
                                                   rmm::cuda_stream_view stream,
                                                   rmm::device_async_resource_ref mr)
{
  auto const output_type = cudf::data_type{cudf::type_to_id<cudf::size_type>()};
  if (input.is_empty()) { return cudf::make_empty_column(output_type); }
  CUDF_EXPECTS(
    input.size() > 1, "the input strings must include at least 2 strings", std::invalid_argument);
  CUDF_EXPECTS(input.size() * static_cast<size_t>(input.size()) <
                 static_cast<std::size_t>(std::numeric_limits<cudf::size_type>().max()),
               "too many strings to create the output column",
               std::overflow_error);

  // create device column of the input strings column
  auto d_strings = cudf::column_device_view::create(input.parent(), stream);

  // Calculate the size of the compute-buffer.
  // We only need memory for half the size of the output matrix since the edit distance calculation
  // is commutative -- `distance(strings[i],strings[j]) == distance(strings[j],strings[i])`
  auto const n_upper     = (input.size() * (input.size() - 1L)) / 2L;
  auto const output_size = input.size() * input.size();
  rmm::device_uvector<std::ptrdiff_t> offsets(n_upper + 1, stream);
  thrust::uninitialized_fill(rmm::exec_policy_nosync(stream), offsets.begin(), offsets.end(), 0);
  thrust::for_each_n(rmm::exec_policy_nosync(stream),
                     thrust::counting_iterator<cudf::size_type>(0),
                     output_size,
                     calculate_matrix_compute_buffer_fn{*d_strings, offsets.data()});

  // get the total size for the compute buffer
  // and convert sizes to offsets in-place
  auto const compute_size =
    cudf::detail::sizes_to_offsets(offsets.begin(), offsets.end(), offsets.begin(), 0, stream);

  // create the compute buffer
  rmm::device_uvector<cudf::size_type> compute_buffer(compute_size, stream);
  auto d_buffer = compute_buffer.data();

  // compute the edit distance into the output column
  auto results = cudf::make_fixed_width_column(
    output_type, output_size, rmm::device_buffer{0, stream, mr}, 0, stream, mr);
  auto d_results = results->mutable_view().data<cudf::size_type>();
  thrust::for_each_n(
    rmm::exec_policy_nosync(stream),
    thrust::counting_iterator<cudf::size_type>(0),
    output_size,
    edit_distance_matrix_levenshtein_algorithm{*d_strings, d_buffer, offsets.data(), d_results});

  // build a lists column of the results
  auto offsets_column =
    cudf::detail::sequence(input.size() + 1,
                           cudf::numeric_scalar<cudf::size_type>(0, true, stream),
                           cudf::numeric_scalar<cudf::size_type>(input.size(), true, stream),
                           stream,
                           mr);
  return cudf::make_lists_column(input.size(),
                                 std::move(offsets_column),
                                 std::move(results),
                                 0,  // no nulls
                                 rmm::device_buffer{0, stream, mr},
                                 stream,
                                 mr);
}

}  // namespace detail

// external APIs

/**
 * @copydoc nvtext::edit_distance
 */
std::unique_ptr<cudf::column> edit_distance(cudf::strings_column_view const& input,
                                            cudf::strings_column_view const& targets,
                                            rmm::cuda_stream_view stream,
                                            rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  return detail::edit_distance(input, targets, stream, mr);
}

/**
 * @copydoc nvtext::edit_distance_matrix
 */
std::unique_ptr<cudf::column> edit_distance_matrix(cudf::strings_column_view const& input,
                                                   rmm::cuda_stream_view stream,
                                                   rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  return detail::edit_distance_matrix(input, stream, mr);
}

}  // namespace nvtext
