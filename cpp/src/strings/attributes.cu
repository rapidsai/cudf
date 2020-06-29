/*
 * Copyright (c) 2019-2020, NVIDIA CORPORATION.
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
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/strings/attributes.hpp>
#include <cudf/strings/string_view.cuh>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/utilities/error.hpp>

#include <rmm/thrust_rmm_allocator.h>
#include <thrust/transform.h>
#include <thrust/transform_scan.h>

namespace cudf {
namespace strings {
namespace detail {
namespace {
/**
 * @brief Returns a numeric column containing lengths of each string in
 * based on the provided unary function.
 *
 * Any null string will result in a null entry for that row in the output column.
 *
 * @tparam UnaryFunction Device function that returns an integer given a string_view.
 * @param strings Strings instance for this operation.
 * @param ufn Function returns an integer for each string.
 * @param stream CUDA stream used for device memory operations and kernel launches.
 * @param mr Device memory resource used to allocate the returned column's device memory
 * @return New INT32 column with lengths for each string.
 */
template <typename UnaryFunction>
std::unique_ptr<column> counts_fn(strings_column_view const& strings,
                                  UnaryFunction& ufn,
                                  rmm::mr::device_memory_resource* mr,
                                  cudaStream_t stream = 0)
{
  auto strings_count  = strings.size();
  auto execpol        = rmm::exec_policy(stream);
  auto strings_column = cudf::column_device_view::create(strings.parent(), stream);
  auto d_strings      = *strings_column;
  // create output column
  auto results = std::make_unique<cudf::column>(
    cudf::data_type{type_id::INT32},
    strings_count,
    rmm::device_buffer(strings_count * sizeof(int32_t), stream, mr),
    copy_bitmask(strings.parent(), stream, mr),  // copy the null mask
    strings.null_count());
  auto results_view = results->mutable_view();
  auto d_lengths    = results_view.data<int32_t>();
  // fill in the lengths
  thrust::transform(execpol->on(stream),
                    thrust::make_counting_iterator<cudf::size_type>(0),
                    thrust::make_counting_iterator<cudf::size_type>(strings_count),
                    d_lengths,
                    [d_strings, ufn] __device__(size_type idx) {
                      int32_t length = 0;
                      if (!d_strings.is_null(idx))
                        length = static_cast<int32_t>(ufn(d_strings.element<string_view>(idx)));
                      return length;
                    });
  results->set_null_count(strings.null_count());  // reset null count
  return results;
}

}  // namespace

std::unique_ptr<column> count_characters(
  strings_column_view const& strings,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource(),
  cudaStream_t stream                 = 0)
{
  auto ufn = [] __device__(const string_view& d_str) { return d_str.length(); };
  return counts_fn(strings, ufn, mr, stream);
}

std::unique_ptr<column> count_bytes(
  strings_column_view const& strings,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource(),
  cudaStream_t stream                 = 0)
{
  auto ufn = [] __device__(const string_view& d_str) { return d_str.size_bytes(); };
  return counts_fn(strings, ufn, mr, stream);
}

}  // namespace detail

namespace {
/**
 * @brief Sets the code-point values for each character in the output
 * integer memory for each string in the strings column.
 *
 * For each string, there is a sub-array in d_results with length equal
 * to the number of characters in that string. The function here will
 * write code-point values to that section as pointed to by the
 * corresponding d_offsets value calculated for that string.
 */
struct code_points_fn {
  column_device_view d_strings;
  size_type* d_offsets;  // offset within d_results to fill with each string's code-point values
  int32_t* d_results;    // base integer array output

  __device__ void operator()(size_type idx)
  {
    if (d_strings.is_null(idx)) return;
    auto d_str  = d_strings.element<string_view>(idx);
    auto result = d_results + d_offsets[idx];
    thrust::copy(thrust::seq, d_str.begin(), d_str.end(), result);
  }
};

}  // namespace

namespace detail {
//
std::unique_ptr<column> code_points(
  strings_column_view const& strings,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource(),
  cudaStream_t stream                 = 0)
{
  auto strings_column = column_device_view::create(strings.parent(), stream);
  auto d_column       = *strings_column;

  // create offsets vector to account for each string's character length
  rmm::device_vector<size_type> offsets(strings.size() + 1);
  size_type* d_offsets = offsets.data().get();
  thrust::transform_inclusive_scan(
    rmm::exec_policy(stream)->on(stream),
    thrust::make_counting_iterator<size_type>(0),
    thrust::make_counting_iterator<size_type>(strings.size()),
    d_offsets + 1,
    [d_column] __device__(size_type idx) {
      size_type length = 0;
      if (!d_column.is_null(idx)) length = d_column.element<string_view>(idx).length();
      return length;
    },
    thrust::plus<size_type>());
  CUDA_TRY(cudaMemsetAsync(d_offsets, 0, sizeof(size_type), stream));

  // the total size is the number of characters in the entire column
  size_type num_characters = offsets.back();
  // create output column with no nulls
  auto results = make_numeric_column(
    data_type{type_id::INT32}, num_characters, mask_state::UNALLOCATED, stream, mr);
  auto results_view = results->mutable_view();
  // fill column with character code-point values
  auto d_results = results_view.data<int32_t>();
  // now set the ranges from each strings' character values
  thrust::for_each_n(rmm::exec_policy(stream)->on(stream),
                     thrust::make_counting_iterator<size_type>(0),
                     strings.size(),
                     code_points_fn{d_column, d_offsets, d_results});
  //
  results->set_null_count(0);
  return results;
}

}  // namespace detail

// external APIS

std::unique_ptr<column> count_characters(strings_column_view const& strings,
                                         rmm::mr::device_memory_resource* mr)
{
  CUDF_FUNC_RANGE();
  return detail::count_characters(strings, mr);
}

std::unique_ptr<column> count_bytes(strings_column_view const& strings,
                                    rmm::mr::device_memory_resource* mr)
{
  CUDF_FUNC_RANGE();
  return detail::count_bytes(strings, mr);
}

std::unique_ptr<column> code_points(strings_column_view const& strings,
                                    rmm::mr::device_memory_resource* mr)
{
  CUDF_FUNC_RANGE();
  return detail::code_points(strings, mr);
}

}  // namespace strings
}  // namespace cudf
