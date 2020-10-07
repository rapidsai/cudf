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
#include <cudf/detail/concatenate.cuh>

#include <cudf/column/column.hpp>
#include <cudf/column/column_device_view.cuh>
#include <cudf/detail/copy.hpp>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/detail/utilities/cuda.cuh>
#include <cudf/dictionary/detail/concatenate.hpp>
#include <cudf/lists/detail/concatenate.hpp>
#include <cudf/strings/detail/concatenate.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_device_view.cuh>

#include <thrust/binary_search.h>
#include <thrust/transform_scan.h>

#include <algorithm>
#include <numeric>
#include <utility>

namespace cudf {
namespace detail {
// From benchmark data, the fused kernel optimization appears to perform better
// when there are more than a trivial number of columns, or when the null mask
// can also be computed at the same time
constexpr bool use_fused_kernel_heuristic(bool const has_nulls, size_t const num_columns)
{
  return has_nulls || num_columns > 4;
}

auto create_device_views(std::vector<column_view> const& views, cudaStream_t stream)
{
  // Create device views for each input view
  using CDViewPtr =
    decltype(column_device_view::create(std::declval<column_view>(), std::declval<cudaStream_t>()));
  auto device_view_owners = std::vector<CDViewPtr>(views.size());
  std::transform(
    views.cbegin(), views.cend(), device_view_owners.begin(), [stream](auto const& col) {
      // TODO creating this device view can invoke null count computation
      // even though it isn't used. See this issue:
      // https://github.com/rapidsai/cudf/issues/4368
      return column_device_view::create(col, stream);
    });

  // Assemble contiguous array of device views
  auto device_views = thrust::host_vector<column_device_view>();
  device_views.reserve(views.size());
  std::transform(device_view_owners.cbegin(),
                 device_view_owners.cend(),
                 std::back_inserter(device_views),
                 [](auto const& col) { return *col; });
  // TODO each of these device vector copies invoke stream synchronization
  // which appears to add unnecessary overhead. See this issue:
  // https://github.com/rapidsai/rmm/issues/120
  auto d_views = rmm::device_vector<column_device_view>{device_views};

  // Compute the partition offsets
  auto offsets = thrust::host_vector<size_t>(views.size() + 1);
  thrust::transform_inclusive_scan(
    thrust::host,
    device_views.cbegin(),
    device_views.cend(),
    std::next(offsets.begin()),
    [](auto const& col) { return col.size(); },
    thrust::plus<size_t>{});
  auto const d_offsets   = rmm::device_vector<size_t>{offsets};
  auto const output_size = offsets.back();

  return std::make_tuple(
    std::move(device_view_owners), std::move(d_views), std::move(d_offsets), output_size);
}

/**
 * @brief Concatenates the null mask bits of all the column device views in the
 * `views` array to the destination bitmask.
 *
 * @param views Array of column_device_view
 * @param output_offsets Prefix sum of sizes of elements of `views`
 * @param number_of_views Size of `views` array
 * @param dest_mask The output buffer to copy null masks into
 * @param number_of_mask_bits The total number of null masks bits that are being
 * copied
 **/
__global__ void concatenate_masks_kernel(column_device_view const* views,
                                         size_t const* output_offsets,
                                         size_type number_of_views,
                                         bitmask_type* dest_mask,
                                         size_type number_of_mask_bits)
{
  size_type mask_index = threadIdx.x + blockIdx.x * blockDim.x;

  auto active_mask = __ballot_sync(0xFFFF'FFFF, mask_index < number_of_mask_bits);

  while (mask_index < number_of_mask_bits) {
    size_type const source_view_index =
      thrust::upper_bound(
        thrust::seq, output_offsets, output_offsets + number_of_views, mask_index) -
      output_offsets - 1;
    bool bit_is_set = 1;
    if (source_view_index < number_of_views) {
      size_type const column_element_index = mask_index - output_offsets[source_view_index];
      bit_is_set = views[source_view_index].is_valid(column_element_index);
    }
    bitmask_type const new_word = __ballot_sync(active_mask, bit_is_set);

    if (threadIdx.x % detail::warp_size == 0) { dest_mask[word_index(mask_index)] = new_word; }

    mask_index += blockDim.x * gridDim.x;
    active_mask = __ballot_sync(active_mask, mask_index < number_of_mask_bits);
  }
}

void concatenate_masks(rmm::device_vector<column_device_view> const& d_views,
                       rmm::device_vector<size_t> const& d_offsets,
                       bitmask_type* dest_mask,
                       size_type output_size,
                       cudaStream_t stream)
{
  constexpr size_type block_size{256};
  cudf::detail::grid_1d config(output_size, block_size);
  concatenate_masks_kernel<<<config.num_blocks, config.num_threads_per_block, 0, stream>>>(
    d_views.data().get(),
    d_offsets.data().get(),
    static_cast<size_type>(d_views.size()),
    dest_mask,
    output_size);
}

void concatenate_masks(std::vector<column_view> const& views,
                       bitmask_type* dest_mask,
                       cudaStream_t stream)
{
  // Preprocess and upload inputs to device memory
  auto const device_views = create_device_views(views, stream);
  auto const& d_views     = std::get<1>(device_views);
  auto const& d_offsets   = std::get<2>(device_views);
  auto const output_size  = std::get<3>(device_views);

  concatenate_masks(d_views, d_offsets, dest_mask, output_size, stream);
}

template <typename T, size_type block_size, bool Nullable>
__global__ void fused_concatenate_kernel(column_device_view const* input_views,
                                         size_t const* input_offsets,
                                         size_type num_input_views,
                                         mutable_column_device_view output_view,
                                         size_type* out_valid_count)
{
  auto const output_size = output_view.size();
  auto* output_data      = output_view.data<T>();

  size_type output_index     = threadIdx.x + blockIdx.x * blockDim.x;
  size_type warp_valid_count = 0;

  unsigned active_mask;
  if (Nullable) { active_mask = __ballot_sync(0xFFFF'FFFF, output_index < output_size); }
  while (output_index < output_size) {
    // Lookup input index by searching for output index in offsets
    // thrust::prev isn't in CUDA 10.0, so subtracting 1 here instead
    auto const offset_it =
      -1 + thrust::upper_bound(
             thrust::seq, input_offsets, input_offsets + num_input_views, output_index);
    size_type const partition_index = offset_it - input_offsets;

    // Copy input data to output
    auto const offset_index   = output_index - *offset_it;
    auto const& input_view    = input_views[partition_index];
    auto const* input_data    = input_view.data<T>();
    output_data[output_index] = input_data[offset_index];

    if (Nullable) {
      bool const bit_is_set       = input_view.is_valid(offset_index);
      bitmask_type const new_word = __ballot_sync(active_mask, bit_is_set);

      // First thread writes bitmask word
      if (threadIdx.x % detail::warp_size == 0) {
        output_view.null_mask()[word_index(output_index)] = new_word;
      }

      warp_valid_count += __popc(new_word);
    }

    output_index += blockDim.x * gridDim.x;
    if (Nullable) { active_mask = __ballot_sync(active_mask, output_index < output_size); }
  }

  if (Nullable) {
    using detail::single_lane_block_sum_reduce;
    auto block_valid_count = single_lane_block_sum_reduce<block_size, 0>(warp_valid_count);
    if (threadIdx.x == 0) { atomicAdd(out_valid_count, block_valid_count); }
  }
}

template <typename T>
std::unique_ptr<column> fused_concatenate(std::vector<column_view> const& views,
                                          bool const has_nulls,
                                          rmm::mr::device_memory_resource* mr,
                                          cudaStream_t stream)
{
  using mask_policy = cudf::mask_allocation_policy;

  // Preprocess and upload inputs to device memory
  auto const device_views = create_device_views(views, stream);
  auto const& d_views     = std::get<1>(device_views);
  auto const& d_offsets   = std::get<2>(device_views);
  auto const output_size  = std::get<3>(device_views);

  CUDF_EXPECTS(output_size < std::numeric_limits<size_type>::max(),
               "Total number of concatenated rows exceeds size_type range");

  // Allocate output
  auto const policy = has_nulls ? mask_policy::ALWAYS : mask_policy::NEVER;
  auto out_col      = detail::allocate_like(views.front(), output_size, policy, mr, stream);
  out_col->set_null_count(0);  // prevent null count from being materialized
  auto out_view   = out_col->mutable_view();
  auto d_out_view = mutable_column_device_view::create(out_view, stream);

  rmm::device_scalar<size_type> d_valid_count(0);

  // Launch kernel
  constexpr size_type block_size{256};
  cudf::detail::grid_1d config(output_size, block_size);
  auto const kernel = has_nulls ? fused_concatenate_kernel<T, block_size, true>
                                : fused_concatenate_kernel<T, block_size, false>;
  kernel<<<config.num_blocks, config.num_threads_per_block, 0, stream>>>(
    d_views.data().get(),
    d_offsets.data().get(),
    static_cast<size_type>(d_views.size()),
    *d_out_view,
    d_valid_count.data());

  if (has_nulls) { out_col->set_null_count(output_size - d_valid_count.value(stream)); }

  return out_col;
}

template <typename T>
std::unique_ptr<column> for_each_concatenate(std::vector<column_view> const& views,
                                             bool const has_nulls,
                                             rmm::mr::device_memory_resource* mr,
                                             cudaStream_t stream)
{
  size_type const total_element_count =
    std::accumulate(views.begin(), views.end(), 0, [](auto accumulator, auto const& v) {
      return accumulator + v.size();
    });

  using mask_policy = cudf::mask_allocation_policy;
  auto const policy = has_nulls ? mask_policy::ALWAYS : mask_policy::NEVER;
  auto col          = cudf::allocate_like(views.front(), total_element_count, policy, mr);

  col->set_null_count(0);             // prevent null count from being materialized...
  auto m_view = col->mutable_view();  // ...when we take a mutable view

  auto count = 0;
  for (auto& v : views) {
    thrust::copy(
      rmm::exec_policy()->on(stream), v.begin<T>(), v.end<T>(), m_view.begin<T>() + count);
    count += v.size();
  }

  // If concatenated column is nullable, proceed to calculate it
  if (has_nulls) {
    cudf::detail::concatenate_masks(views, (col->mutable_view()).null_mask(), stream);
  }

  return col;
}

struct concatenate_dispatch {
  std::vector<column_view> const& views;
  rmm::mr::device_memory_resource* mr;
  cudaStream_t stream;

  // fixed width
  template <typename T>
  std::unique_ptr<column> operator()()
  {
    bool const has_nulls =
      std::any_of(views.cbegin(), views.cend(), [](auto const& col) { return col.has_nulls(); });

    using Type = device_storage_type_t<T>;

    // Use a heuristic to guess when the fused kernel will be faster
    if (use_fused_kernel_heuristic(has_nulls, views.size())) {
      return fused_concatenate<Type>(views, has_nulls, mr, stream);
    } else {
      return for_each_concatenate<Type>(views, has_nulls, mr, stream);
    }
  }
};

template <>
std::unique_ptr<column> concatenate_dispatch::operator()<cudf::dictionary32>()
{
  return cudf::dictionary::detail::concatenate(views, stream, mr);
}

template <>
std::unique_ptr<column> concatenate_dispatch::operator()<cudf::string_view>()
{
  return cudf::strings::detail::concatenate(views, mr, stream);
}

template <>
std::unique_ptr<column> concatenate_dispatch::operator()<cudf::list_view>()
{
  return cudf::lists::detail::concatenate(views, stream, mr);
}

// Concatenates the elements from a vector of column_views
std::unique_ptr<column> concatenate(std::vector<column_view> const& columns_to_concat,
                                    rmm::mr::device_memory_resource* mr,
                                    cudaStream_t stream)
{
  CUDF_EXPECTS(not columns_to_concat.empty(), "Unexpected empty list of columns to concatenate.");

  data_type const type = columns_to_concat.front().type();
  CUDF_EXPECTS(std::all_of(columns_to_concat.begin(),
                           columns_to_concat.end(),
                           [&type](auto const& c) { return c.type() == type; }),
               "Type mismatch in columns to concatenate.");

  if (std::all_of(columns_to_concat.begin(), columns_to_concat.end(), [](column_view const& c) {
        return c.is_empty();
      })) {
    return empty_like(columns_to_concat.front());
  }

  return type_dispatcher(type, concatenate_dispatch{columns_to_concat, mr, stream});
}

std::unique_ptr<table> concatenate(std::vector<table_view> const& tables_to_concat,
                                   rmm::mr::device_memory_resource* mr,
                                   cudaStream_t stream)
{
  if (tables_to_concat.empty()) { return std::make_unique<table>(); }

  table_view const first_table = tables_to_concat.front();
  CUDF_EXPECTS(std::all_of(tables_to_concat.cbegin(),
                           tables_to_concat.cend(),
                           [&first_table](auto const& t) {
                             return t.num_columns() == first_table.num_columns() &&
                                    have_same_types(first_table, t);
                           }),
               "Mismatch in table columns to concatenate.");

  std::vector<std::unique_ptr<column>> concat_columns;
  for (size_type i = 0; i < first_table.num_columns(); ++i) {
    std::vector<column_view> cols;
    std::transform(tables_to_concat.cbegin(),
                   tables_to_concat.cend(),
                   std::back_inserter(cols),
                   [i](auto const& t) { return t.column(i); });
    concat_columns.emplace_back(detail::concatenate(cols, mr, stream));
  }
  return std::make_unique<table>(std::move(concat_columns));
}

}  // namespace detail

rmm::device_buffer concatenate_masks(std::vector<column_view> const& views,
                                     rmm::mr::device_memory_resource* mr)
{
  bool const has_nulls =
    std::any_of(views.begin(), views.end(), [](const column_view col) { return col.has_nulls(); });
  if (has_nulls) {
    size_type const total_element_count =
      std::accumulate(views.begin(), views.end(), 0, [](auto accumulator, auto const& v) {
        return accumulator + v.size();
      });

    rmm::device_buffer null_mask =
      create_null_mask(total_element_count, mask_state::UNINITIALIZED, 0, mr);

    detail::concatenate_masks(views, static_cast<bitmask_type*>(null_mask.data()), 0);

    return null_mask;
  }
  // no nulls, so return an empty device buffer
  return rmm::device_buffer{0, (cudaStream_t)0, mr};
}

// Concatenates the elements from a vector of column_views
std::unique_ptr<column> concatenate(std::vector<column_view> const& columns_to_concat,
                                    rmm::mr::device_memory_resource* mr)
{
  CUDF_FUNC_RANGE();
  return detail::concatenate(columns_to_concat, mr, 0);
}

std::unique_ptr<table> concatenate(std::vector<table_view> const& tables_to_concat,
                                   rmm::mr::device_memory_resource* mr)
{
  CUDF_FUNC_RANGE();
  return detail::concatenate(tables_to_concat, mr, 0);
}

}  // namespace cudf
