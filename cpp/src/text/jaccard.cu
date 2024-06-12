/*
 * Copyright (c) 2023-2024, NVIDIA CORPORATION.
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
#include <cudf/detail/null_mask.hpp>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/detail/utilities/cuda.cuh>
#include <cudf/lists/lists_column_view.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/span.hpp>

#include <nvtext/detail/generate_ngrams.hpp>
#include <nvtext/jaccard.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/exec_policy.hpp>
#include <rmm/resource_ref.hpp>

#include <cub/cub.cuh>
#include <thrust/binary_search.h>
#include <thrust/execution_policy.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/transform.h>

namespace nvtext {
namespace detail {
namespace {

/**
 * @brief Retrieve the row data (span) for the given column/row-index
 *
 * @param d_input Input lists column
 * @param idx Row index to retrieve
 * @return A device-span of the row values
 */
__device__ auto get_row(cudf::column_device_view const& d_input, cudf::size_type idx)
{
  auto const offsets =
    d_input.child(cudf::lists_column_view::offsets_column_index).data<cudf::size_type>();
  auto const offset = offsets[idx];
  auto const size   = offsets[idx + 1] - offset;
  auto const begin =
    d_input.child(cudf::lists_column_view::child_column_index).data<uint32_t>() + offset;
  return cudf::device_span<uint32_t const>(begin, size);
}

/**
 * @brief Count the unique values within each row of the input column
 *
 * This is called with a warp per row
 */
struct sorted_unique_fn {
  cudf::column_device_view const d_input;
  cudf::size_type* d_results;

  // warp per row
  __device__ void operator()(cudf::size_type idx) const
  {
    using warp_reduce = cub::WarpReduce<cudf::size_type>;
    __shared__ typename warp_reduce::TempStorage temp_storage;

    auto const row_idx  = idx / cudf::detail::warp_size;
    auto const lane_idx = idx % cudf::detail::warp_size;
    auto const row      = get_row(d_input, row_idx);
    auto const begin    = row.begin();

    cudf::size_type count = 0;
    for (auto itr = begin + lane_idx; itr < row.end(); itr += cudf::detail::warp_size) {
      count += (itr == begin || *itr != *(itr - 1));
    }
    auto const result = warp_reduce(temp_storage).Sum(count);
    if (lane_idx == 0) { d_results[row_idx] = result; }
  }
};

rmm::device_uvector<cudf::size_type> compute_unique_counts(cudf::column_view const& input,
                                                           rmm::cuda_stream_view stream)
{
  auto const d_input = cudf::column_device_view::create(input, stream);
  auto d_results     = rmm::device_uvector<cudf::size_type>(input.size(), stream);
  sorted_unique_fn fn{*d_input, d_results.data()};
  thrust::for_each_n(rmm::exec_policy(stream),
                     thrust::counting_iterator<cudf::size_type>(0),
                     input.size() * cudf::detail::warp_size,
                     fn);
  return d_results;
}

/**
 * @brief Count the number of common values within each row of the 2 input columns
 *
 * This is called with a warp per row
 */
struct sorted_intersect_fn {
  cudf::column_device_view const d_input1;
  cudf::column_device_view const d_input2;
  cudf::size_type* d_results;

  // warp per row
  __device__ void operator()(cudf::size_type idx) const
  {
    using warp_reduce = cub::WarpReduce<cudf::size_type>;
    __shared__ typename warp_reduce::TempStorage temp_storage;

    auto const row_idx  = idx / cudf::detail::warp_size;
    auto const lane_idx = idx % cudf::detail::warp_size;

    auto const needles  = get_row(d_input1, row_idx);
    auto const haystack = get_row(d_input2, row_idx);

    auto begin     = haystack.begin();
    auto const end = haystack.end();

    // TODO: investigate cuCollections device-side static-map to match row values

    cudf::size_type count = 0;
    for (auto itr = needles.begin() + lane_idx; itr < needles.end() && begin < end;
         itr += cudf::detail::warp_size) {
      if (itr != needles.begin() && *itr == *(itr - 1)) { continue; }  // skip duplicates
      // search haystack for this needle (*itr)
      auto const found = thrust::lower_bound(thrust::seq, begin, end, *itr);
      count += (found != end) && (*found == *itr);  // increment if found;
      begin = found;                                // shorten the next lower-bound range
    }
    // sum up the counts across this warp
    auto const result = warp_reduce(temp_storage).Sum(count);
    if (lane_idx == 0) { d_results[row_idx] = result; }
  }
};

rmm::device_uvector<cudf::size_type> compute_intersect_counts(cudf::column_view const& input1,
                                                              cudf::column_view const& input2,
                                                              rmm::cuda_stream_view stream)
{
  auto const d_input1 = cudf::column_device_view::create(input1, stream);
  auto const d_input2 = cudf::column_device_view::create(input2, stream);
  auto d_results      = rmm::device_uvector<cudf::size_type>(input1.size(), stream);
  sorted_intersect_fn fn{*d_input1, *d_input2, d_results.data()};
  thrust::for_each_n(rmm::exec_policy(stream),
                     thrust::counting_iterator<cudf::size_type>(0),
                     input1.size() * cudf::detail::warp_size,
                     fn);
  return d_results;
}

/**
 * @brief Compute the jaccard distance for each row
 *
 * Formula is J = |A ∩ B| / |A ∪ B|
 *              = |A ∩ B| / (|A| + |B| - |A ∩ B|)
 *
 * where |A ∩ B| is number of common values between A and B
 * and |x| is the number of unique values in x.
 */
struct jaccard_fn {
  cudf::size_type const* d_uniques1;
  cudf::size_type const* d_uniques2;
  cudf::size_type const* d_intersects;

  __device__ float operator()(cudf::size_type idx) const
  {
    auto const count1     = d_uniques1[idx];
    auto const count2     = d_uniques2[idx];
    auto const intersects = d_intersects[idx];
    // the intersect values are in both sets so a union count
    // would need to subtract the intersect count from one set
    // (see formula in comment above)
    auto const unions = count1 + count2 - intersects;
    return unions ? (static_cast<float>(intersects) / static_cast<float>(unions)) : 0.f;
  }
};

/**
 * @brief Create hashes for each substring
 *
 * Uses the hash_character_ngrams to hash substrings of the input column.
 * This returns a lists column where each row is the hashes for the substrings
 * of the corresponding input string row.
 *
 * The hashes are then sorted using a segmented-sort as setup to
 * perform the unique and intersect operations.
 */
std::unique_ptr<cudf::column> hash_substrings(cudf::strings_column_view const& col,
                                              cudf::size_type width,
                                              rmm::cuda_stream_view stream)
{
  auto hashes = hash_character_ngrams(col, width, stream, rmm::mr::get_current_device_resource());
  auto const input   = cudf::lists_column_view(hashes->view());
  auto const offsets = input.offsets_begin();
  auto const data    = input.child().data<uint32_t>();

  rmm::device_uvector<uint32_t> sorted(input.child().size(), stream);

  // this is wicked fast and much faster than using cudf::lists::detail::sort_list
  rmm::device_buffer d_temp_storage;
  size_t temp_storage_bytes = 0;
  cub::DeviceSegmentedSort::SortKeys(d_temp_storage.data(),
                                     temp_storage_bytes,
                                     data,
                                     sorted.data(),
                                     sorted.size(),
                                     input.size(),
                                     offsets,
                                     offsets + 1,
                                     stream.value());
  d_temp_storage = rmm::device_buffer{temp_storage_bytes, stream};
  cub::DeviceSegmentedSort::SortKeys(d_temp_storage.data(),
                                     temp_storage_bytes,
                                     data,
                                     sorted.data(),
                                     sorted.size(),
                                     input.size(),
                                     offsets,
                                     offsets + 1,
                                     stream.value());

  auto contents = hashes->release();
  // the offsets are taken from the hashes column since they are the same
  // before and after the segmented-sort
  return cudf::make_lists_column(
    col.size(),
    std::move(contents.children.front()),
    std::make_unique<cudf::column>(std::move(sorted), rmm::device_buffer{}, 0),
    0,
    rmm::device_buffer{},
    stream,
    rmm::mr::get_current_device_resource());
}
}  // namespace

std::unique_ptr<cudf::column> jaccard_index(cudf::strings_column_view const& input1,
                                            cudf::strings_column_view const& input2,
                                            cudf::size_type width,
                                            rmm::cuda_stream_view stream,
                                            rmm::device_async_resource_ref mr)
{
  CUDF_EXPECTS(
    input1.size() == input2.size(), "input columns must be the same size", std::invalid_argument);
  CUDF_EXPECTS(width >= 2,
               "Parameter width should be an integer value of 2 or greater",
               std::invalid_argument);

  constexpr auto output_type = cudf::data_type{cudf::type_id::FLOAT32};
  if (input1.is_empty()) { return cudf::make_empty_column(output_type); }

  auto const [d_uniques1, d_uniques2, d_intersects] = [&] {
    // build hashes of the substrings
    auto const hash1 = hash_substrings(input1, width, stream);
    auto const hash2 = hash_substrings(input2, width, stream);

    // compute the unique counts in each set and the intersection counts
    auto d_uniques1   = compute_unique_counts(hash1->view(), stream);
    auto d_uniques2   = compute_unique_counts(hash2->view(), stream);
    auto d_intersects = compute_intersect_counts(hash1->view(), hash2->view(), stream);

    return std::tuple{std::move(d_uniques1), std::move(d_uniques2), std::move(d_intersects)};
  }();

  auto results = cudf::make_numeric_column(
    output_type, input1.size(), cudf::mask_state::UNALLOCATED, stream, mr);
  auto d_results = results->mutable_view().data<float>();

  // compute the jaccard using the unique counts and the intersect counts
  thrust::transform(rmm::exec_policy(stream),
                    thrust::counting_iterator<cudf::size_type>(0),
                    thrust::counting_iterator<cudf::size_type>(results->size()),
                    d_results,
                    jaccard_fn{d_uniques1.data(), d_uniques2.data(), d_intersects.data()});

  if (input1.null_count() || input2.null_count()) {
    auto [null_mask, null_count] =
      cudf::detail::bitmask_and(cudf::table_view({input1.parent(), input2.parent()}), stream, mr);
    results->set_null_mask(std::move(null_mask), null_count);
  }

  return results;
}

}  // namespace detail

std::unique_ptr<cudf::column> jaccard_index(cudf::strings_column_view const& input1,
                                            cudf::strings_column_view const& input2,
                                            cudf::size_type width,
                                            rmm::cuda_stream_view stream,
                                            rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  return detail::jaccard_index(input1, input2, width, stream, mr);
}

}  // namespace nvtext
