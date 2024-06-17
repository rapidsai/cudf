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

#include <cudf/column/column.hpp>
#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_factories.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/strings/string_view.cuh>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/utilities/default_stream.hpp>

#include <nvtext/edit_distance.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>
#include <rmm/resource_ref.hpp>

#include <thrust/execution_policy.h>
#include <thrust/for_each.h>
#include <thrust/functional.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/reduce.h>
#include <thrust/scan.h>
#include <thrust/sequence.h>
#include <thrust/transform.h>
#include <thrust/transform_scan.h>

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

  auto begin = str_length < tgt_length ? d_str.begin() : d_tgt.begin();
  auto itr   = str_length < tgt_length ? d_tgt.begin() : d_str.begin();
  // .first is min and .second is max
  auto const [n, m] = std::minmax(str_length, tgt_length);
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
      v1[j + 1]     = std::min(std::min(sub_cost, del_cost), ins_cost);
    }
    thrust::swap(v0, v1);
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
    auto d_tgt = [&] __device__ {  // d_targets is also allowed to have only one entry
      if (d_targets.is_null(idx)) { return cudf::string_view{}; }
      return d_targets.size() == 1 ? d_targets.element<cudf::string_view>(0)
                                   : d_targets.element<cudf::string_view>(idx);
    }();
    d_results[idx] = compute_distance(d_str, d_tgt, d_buffer + d_offsets[idx]);
  }
};

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
    auto work_buffer    = d_buffer + d_offsets[idx - ((row + 1) * (row + 2)) / 2];
    auto const distance = (row == col) ? 0 : compute_distance(d_str1, d_str2, work_buffer);
    d_results[idx]      = distance;                   // top half of matrix
    d_results[col * strings_count + row] = distance;  // bottom half of matrix
  }
};

}  // namespace

/**
 * @copydoc nvtext::edit_distance
 */
std::unique_ptr<cudf::column> edit_distance(cudf::strings_column_view const& strings,
                                            cudf::strings_column_view const& targets,
                                            rmm::cuda_stream_view stream,
                                            rmm::device_async_resource_ref mr)
{
  auto const strings_count = strings.size();
  if (strings_count == 0) {
    return cudf::make_empty_column(cudf::data_type{cudf::type_to_id<cudf::size_type>()});
  }
  if (targets.size() > 1) {
    CUDF_EXPECTS(strings_count == targets.size(), "targets.size() must equal strings.size()");
  }

  // create device columns from the input columns
  auto strings_column = cudf::column_device_view::create(strings.parent(), stream);
  auto d_strings      = *strings_column;
  auto targets_column = cudf::column_device_view::create(targets.parent(), stream);
  auto d_targets      = *targets_column;

  // calculate the size of the compute-buffer;
  rmm::device_uvector<std::ptrdiff_t> offsets(strings_count, stream);
  thrust::transform(rmm::exec_policy(stream),
                    thrust::make_counting_iterator<cudf::size_type>(0),
                    thrust::make_counting_iterator<cudf::size_type>(strings_count),
                    offsets.begin(),
                    [d_strings, d_targets] __device__(auto idx) {
                      if (d_strings.is_null(idx) || d_targets.is_null(idx)) {
                        return cudf::size_type{0};
                      }
                      auto d_str = d_strings.element<cudf::string_view>(idx);
                      auto d_tgt = d_targets.size() == 1
                                     ? d_targets.element<cudf::string_view>(0)
                                     : d_targets.element<cudf::string_view>(idx);
                      // just need 2 integers for each character of the shorter string
                      return (std::min(d_str.length(), d_tgt.length()) + 1) * 2;
                    });

  // get the total size of the temporary compute buffer
  int64_t compute_size =
    thrust::reduce(rmm::exec_policy(stream), offsets.begin(), offsets.end(), int64_t{0});
  // convert sizes to offsets in-place
  thrust::exclusive_scan(rmm::exec_policy(stream), offsets.begin(), offsets.end(), offsets.begin());
  // create the temporary compute buffer
  rmm::device_uvector<cudf::size_type> compute_buffer(compute_size, stream);
  auto d_buffer = compute_buffer.data();

  auto results = cudf::make_fixed_width_column(cudf::data_type{cudf::type_to_id<cudf::size_type>()},
                                               strings_count,
                                               rmm::device_buffer{0, stream, mr},
                                               0,
                                               stream,
                                               mr);
  auto d_results = results->mutable_view().data<cudf::size_type>();

  // compute the edit distance into the output column
  thrust::for_each_n(
    rmm::exec_policy(stream),
    thrust::make_counting_iterator<cudf::size_type>(0),
    strings_count,
    edit_distance_levenshtein_algorithm{d_strings, d_targets, d_buffer, offsets.data(), d_results});
  return results;
}

/**
 * @copydoc nvtext::edit_distance_matrix
 */
std::unique_ptr<cudf::column> edit_distance_matrix(cudf::strings_column_view const& strings,
                                                   rmm::cuda_stream_view stream,
                                                   rmm::device_async_resource_ref mr)
{
  cudf::size_type strings_count = strings.size();
  if (strings_count == 0) {
    return cudf::make_empty_column(cudf::data_type{cudf::type_to_id<cudf::size_type>()});
  }
  CUDF_EXPECTS(strings_count > 1, "the input strings must include at least 2 strings");
  CUDF_EXPECTS(static_cast<size_t>(strings_count) * static_cast<size_t>(strings_count) <
                 static_cast<std::size_t>(std::numeric_limits<cudf::size_type>().max()),
               "too many strings to create the output column");

  // create device column of the input strings column
  auto strings_column = cudf::column_device_view::create(strings.parent(), stream);
  auto d_strings      = *strings_column;

  // Calculate the size of the compute-buffer.
  // We only need memory for half the size of the output matrix since the edit distance calculation
  // is commutative -- `distance(strings[i],strings[j]) == distance(strings[j],strings[i])`
  cudf::size_type n_upper = (strings_count * (strings_count - 1)) / 2;
  rmm::device_uvector<std::ptrdiff_t> offsets(n_upper, stream);
  auto d_offsets = offsets.data();
  CUDF_CUDA_TRY(cudaMemsetAsync(d_offsets, 0, n_upper * sizeof(std::ptrdiff_t), stream.value()));
  thrust::for_each_n(
    rmm::exec_policy(stream),
    thrust::make_counting_iterator<cudf::size_type>(0),
    strings_count * strings_count,
    [d_strings, d_offsets, strings_count] __device__(cudf::size_type idx) {
      auto const row = idx / strings_count;
      auto const col = idx % strings_count;
      if (row >= col) return;  // compute only the top half
      cudf::string_view const d_str1 =
        d_strings.is_null(row) ? cudf::string_view{} : d_strings.element<cudf::string_view>(row);
      cudf::string_view const d_str2 =
        d_strings.is_null(col) ? cudf::string_view{} : d_strings.element<cudf::string_view>(col);
      if (d_str1.empty() || d_str2.empty()) { return; }
      // the temp size needed is 2 integers per character of the shorter string
      d_offsets[idx - ((row + 1) * (row + 2)) / 2] =
        (std::min(d_str1.length(), d_str2.length()) + 1) * 2;
    });

  // get the total size for the compute buffer
  int64_t compute_size =
    thrust::reduce(rmm::exec_policy(stream), offsets.begin(), offsets.end(), int64_t{0});
  // convert sizes to offsets in-place
  thrust::exclusive_scan(rmm::exec_policy(stream), offsets.begin(), offsets.end(), offsets.begin());
  // create the compute buffer
  rmm::device_uvector<cudf::size_type> compute_buffer(compute_size, stream);
  auto d_buffer = compute_buffer.data();

  // compute the edit distance into the output column
  auto results = cudf::make_fixed_width_column(cudf::data_type{cudf::type_to_id<cudf::size_type>()},
                                               strings_count * strings_count,
                                               rmm::device_buffer{0, stream, mr},
                                               0,
                                               stream,
                                               mr);
  auto d_results = results->mutable_view().data<cudf::size_type>();
  thrust::for_each_n(
    rmm::exec_policy(stream),
    thrust::make_counting_iterator<cudf::size_type>(0),
    strings_count * strings_count,
    edit_distance_matrix_levenshtein_algorithm{d_strings, d_buffer, d_offsets, d_results});

  // build a lists column of the results
  auto offsets_column =
    cudf::make_fixed_width_column(cudf::data_type{cudf::type_to_id<cudf::size_type>()},
                                  strings_count + 1,
                                  rmm::device_buffer{0, stream, mr},
                                  0,
                                  stream,
                                  mr);
  thrust::transform_exclusive_scan(
    rmm::exec_policy(stream),
    thrust::counting_iterator<cudf::size_type>(0),
    thrust::counting_iterator<cudf::size_type>(strings_count + 1),
    offsets_column->mutable_view().data<cudf::size_type>(),
    [strings_count] __device__(auto idx) { return strings_count; },
    cudf::size_type{0},
    thrust::plus<cudf::size_type>());
  return cudf::make_lists_column(strings_count,
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
