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
#include <cudf/strings/string_view.cuh>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <nvtext/edit_distance.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

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

struct edit_distance_levenshtein_algorithm {
  cudf::column_device_view d_strings;  // computing these
  cudf::column_device_view d_targets;  // against these;
  cudf::size_type* d_buffer;           // compute buffer for each string
  std::ptrdiff_t const* d_offsets;     // locate sub-buffer for each string
  cudf::size_type* d_results;          // edit distance values

  __device__ void operator()(cudf::size_type idx) const
  {
    auto d_str =
      d_strings.is_null(idx) ? cudf::string_view{} : d_strings.element<cudf::string_view>(idx);
    auto d_tgt = [&] __device__ {  // d_targets is also allowed to have only one valid entry
      if (d_targets.size() > 1 && d_targets.is_null(idx)) { return cudf::string_view{}; }
      return d_targets.size() == 1 ? d_targets.element<cudf::string_view>(0)
                                   : d_targets.element<cudf::string_view>(idx);
    }();
    d_results[idx] = compute_distance(d_str, d_tgt, d_buffer + d_offsets[idx]);
  }
};

struct calculate_compute_buffer_fn {
  cudf::column_device_view d_strings;
  cudf::column_device_view d_targets;

  __device__ std::ptrdiff_t operator()(cudf::size_type idx) const
  {
    if (d_strings.is_null(idx)) { return 0; }
    if (d_targets.size() > 1 && d_targets.is_null(idx)) { return 0; }
    auto d_str = d_strings.element<cudf::string_view>(idx);
    auto d_tgt = d_targets.size() == 1 ? d_targets.element<cudf::string_view>(0)
                                       : d_targets.element<cudf::string_view>(idx);
    // just need 2 integers for each character of the shorter string
    return (cuda::std::min(d_str.length(), d_tgt.length()) + 1L) * 2L;
  }
};

}  // namespace

/**
 * @copydoc nvtext::edit_distance
 */
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
                    calculate_compute_buffer_fn{*d_strings, *d_targets});

  // get the total size of the temporary compute buffer
  // and convert sizes to offsets in-place
  auto const compute_size =
    cudf::detail::sizes_to_offsets(offsets.begin(), offsets.end(), offsets.begin(), 0, stream);
  rmm::device_uvector<cudf::size_type> compute_buffer(compute_size, stream);
  auto d_buffer = compute_buffer.data();

  auto results = cudf::make_fixed_width_column(
    output_type, input.size(), rmm::device_buffer{0, stream, mr}, 0, stream, mr);
  auto d_results = results->mutable_view().data<cudf::size_type>();

  // compute the edit distance into the output column
  thrust::for_each_n(rmm::exec_policy_nosync(stream),
                     thrust::counting_iterator<cudf::size_type>(0),
                     input.size(),
                     edit_distance_levenshtein_algorithm{
                       *d_strings, *d_targets, d_buffer, offsets.data(), d_results});
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
