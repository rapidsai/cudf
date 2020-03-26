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
#include <cudf/column/column_factories.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_device_view.cuh>
#include <cudf/detail/copy.hpp>
#include <cudf/detail/utilities/cuda.cuh>
#include <cudf/strings/detail/concatenate.hpp>
#include <cudf/detail/nvtx/ranges.hpp>

#include <thrust/binary_search.h>
#include <thrust/transform_scan.h>

#include <algorithm>
#include <numeric>
#include <utility>

namespace cudf {

namespace detail {

auto create_device_views(
    std::vector<column_view> const& views, cudaStream_t stream) {

  // Create device views for each input view
  using CDViewPtr = decltype(column_device_view::create(
      std::declval<column_view>(), std::declval<cudaStream_t>()));
  auto device_view_owners = std::vector<CDViewPtr>(views.size());
  std::transform(views.cbegin(), views.cend(),
      device_view_owners.begin(),
      [stream](auto const& col) {
        // TODO creating this device view can invoke null count computation
        // even though it isn't used. See this issue:
        // https://github.com/rapidsai/cudf/issues/4368
        return column_device_view::create(col, stream);
      });

  // Assemble contiguous array of device views
  auto device_views = thrust::host_vector<column_device_view>();
  device_views.reserve(views.size());
  std::transform(device_view_owners.cbegin(), device_view_owners.cend(),
      std::back_inserter(device_views),
      [](auto const& col) {
        return *col;
      });
  // TODO each of these device vector copies invoke stream synchronization
  // which appears to add unnecessary overhead. See this issue:
  // https://github.com/rapidsai/rmm/issues/120
  auto d_views = rmm::device_vector<column_device_view>{device_views};

  // Compute the partition offsets
  auto offsets = thrust::host_vector<size_type>(views.size() + 1);
  thrust::transform_inclusive_scan(thrust::host,
      device_views.cbegin(), device_views.cend(),
      std::next(offsets.begin()),
      [](auto const& col) {
        return col.size();
      },
      thrust::plus<size_type>{});
  auto const d_offsets = rmm::device_vector<size_type>{offsets};
  auto const output_size = offsets.back();

  return std::make_tuple(
      std::move(device_view_owners),
      std::move(d_views),
      std::move(d_offsets),
      output_size);
}

// Using a functor instead of a lambda as a workaround for:
// error: The enclosing parent function ("create_strings_device_views") for an
// extended __device__ lambda must not have deduced return type
struct chars_size_transform {
  __device__ size_type operator()(column_device_view const& col) const {
    if (col.size() > 0) {
      constexpr auto offsets_index = strings_column_view::offsets_column_index;
      auto d_offsets = col.child(offsets_index).data<int32_t>();
      return d_offsets[col.size() + col.offset()] - d_offsets[col.offset()];
    } else {
      return 0;
    }
  }
};

auto create_strings_device_views(
    std::vector<column_view> const& views, cudaStream_t stream) {

  // Create device views for each input view
  using CDViewPtr = decltype(column_device_view::create(
      std::declval<column_view>(), std::declval<cudaStream_t>()));
  auto device_view_owners = std::vector<CDViewPtr>(views.size());
  std::transform(views.cbegin(), views.cend(),
      device_view_owners.begin(),
      [stream](auto const& col) {
        return column_device_view::create(col, stream);
      });

  // Assemble contiguous array of device views
  auto device_views = thrust::host_vector<column_device_view>();
  device_views.reserve(views.size());
  std::transform(device_view_owners.cbegin(), device_view_owners.cend(),
      std::back_inserter(device_views),
      [](auto const& col) {
        return *col;
      });
  auto d_views = rmm::device_vector<column_device_view>{device_views};

  // Compute the partition offsets and size of offset column
  auto input_offsets = thrust::host_vector<size_type>(views.size() + 1);
  thrust::transform_inclusive_scan(thrust::host,
      device_views.cbegin(), device_views.cend(),
      std::next(input_offsets.begin()),
      [](auto const& col) {
        return col.size();
      },
      thrust::plus<size_type>{});
  auto const d_input_offsets = rmm::device_vector<size_type>{input_offsets};
  auto const output_size = input_offsets.back();

  // Compute the partition offsets and size of chars column
  // Using separate transform and inclusive_scan because
  // transform_inclusive_scan fails to compile with:
  // error: the default constructor of "cudf::column_device_view" cannot be
  // referenced -- it is a deleted function
  auto d_partition_offsets = rmm::device_vector<size_type>(views.size() + 1);
  thrust::transform(rmm::exec_policy(stream)->on(stream),
      d_views.cbegin(), d_views.cend(),
      std::next(d_partition_offsets.begin()),
      chars_size_transform{});
  thrust::inclusive_scan(rmm::exec_policy(stream)->on(stream),
      d_partition_offsets.cbegin(), d_partition_offsets.cend(),
      d_partition_offsets.begin());

  // TODO just copy the last element back to host
  auto const partition_offsets = thrust::host_vector<size_type>{d_partition_offsets};
  auto const output_chars_size = partition_offsets.back();

  return std::make_tuple(
      std::move(device_view_owners),
      std::move(d_views),
      std::move(d_input_offsets),
      std::move(d_partition_offsets),
      output_size,
      output_chars_size);
}

/**---------------------------------------------------------------------------*
 * @brief Concatenates the null mask bits of all the column device views in the
 * `views` array to the destination bitmask.
 *
 * @param views Array of column_device_view
 * @param output_offsets Prefix sum of sizes of elements of `views`
 * @param number_of_views Size of `views` array
 * @param dest_mask The output buffer to copy null masks into
 * @param number_of_mask_bits The total number of null masks bits that are being
 * copied
 *---------------------------------------------------------------------------**/
__global__
void
concatenate_masks_kernel(
    column_device_view const* views,
    size_type const* output_offsets,
    size_type number_of_views,
    bitmask_type* dest_mask,
    size_type number_of_mask_bits) {

  size_type mask_index = threadIdx.x + blockIdx.x * blockDim.x;

  auto active_mask =
      __ballot_sync(0xFFFF'FFFF, mask_index < number_of_mask_bits);

  while (mask_index < number_of_mask_bits) {
    size_type const source_view_index = thrust::upper_bound(thrust::seq,
        output_offsets, output_offsets + number_of_views,
        mask_index) - output_offsets - 1;
    bool bit_is_set = 1;
    if (source_view_index < number_of_views) {
      size_type const column_element_index = mask_index - output_offsets[source_view_index];
      bit_is_set = views[source_view_index].is_valid(column_element_index);
    }
    bitmask_type const new_word = __ballot_sync(active_mask, bit_is_set);

    if (threadIdx.x % experimental::detail::warp_size == 0) {
      dest_mask[word_index(mask_index)] = new_word;
    }

    mask_index += blockDim.x * gridDim.x;
    active_mask =
        __ballot_sync(active_mask, mask_index < number_of_mask_bits);
  }
}

void concatenate_masks(
    rmm::device_vector<column_device_view> const& d_views,
    rmm::device_vector<size_type> const& d_offsets,
    bitmask_type * dest_mask,
    size_type output_size,
    cudaStream_t stream) {

  constexpr size_type block_size{256};
  cudf::experimental::detail::grid_1d config(output_size, block_size);
  concatenate_masks_kernel<<<config.num_blocks, config.num_threads_per_block,
                             0, stream>>>(
    d_views.data().get(),
    d_offsets.data().get(),
    static_cast<size_type>(d_views.size()),
    dest_mask, output_size);
}

void concatenate_masks(std::vector<column_view> const &views,
    bitmask_type * dest_mask,
    cudaStream_t stream) {

  // Preprocess and upload inputs to device memory
  auto const device_views = create_device_views(views, stream);
  auto const& d_views = std::get<1>(device_views);
  auto const& d_offsets = std::get<2>(device_views);
  auto const output_size = std::get<3>(device_views);

  concatenate_masks(d_views, d_offsets, dest_mask, output_size, stream);
}

struct for_each_concatenate {
  std::vector<cudf::column_view> views;
  bool const has_nulls;
  cudaStream_t stream;
  rmm::mr::device_memory_resource *mr;

  template <typename ColumnType,
      std::enable_if_t<std::is_same<ColumnType, cudf::string_view>::value>* = nullptr>
  std::unique_ptr<column> operator()() {
    std::vector<cudf::strings_column_view> sviews;
    sviews.reserve(views.size());
    for (auto &v : views) { sviews.emplace_back(v); }

    auto col = cudf::strings::detail::concatenate(sviews, mr, stream);

    //If concatenated string column is nullable, proceed to calculate it
    if (col->nullable()) {
      cudf::detail::concatenate_masks(views,
          (col->mutable_view()).null_mask(), stream);
    }

    return col;
  }

  template <typename ColumnType,
      std::enable_if_t<std::is_same<ColumnType, cudf::dictionary32>::value>* = nullptr>
  std::unique_ptr<column> operator()() {
    CUDF_FAIL("dictionary not supported yet");
  }

  template <typename ColumnType,
      std::enable_if_t<cudf::is_fixed_width<ColumnType>()>* = nullptr>
  std::unique_ptr<column> operator()() {

    size_type const total_element_count =
      std::accumulate(views.begin(), views.end(), 0,
          [](auto accumulator, auto const& v) { return accumulator + v.size(); });

    using mask_policy = cudf::experimental::mask_allocation_policy;

    mask_policy policy{mask_policy::NEVER};
    if (has_nulls) { policy = mask_policy::ALWAYS; }

    auto col = cudf::experimental::allocate_like(views.front(),
        total_element_count, policy, mr);
    col->set_null_count(0); // prevent null count from being materialized

    auto m_view = col->mutable_view();
    auto count = 0;
    // NOTE fused_concatenate is more efficient for multiple views
    for (auto &v : views) {
      thrust::copy(rmm::exec_policy()->on(stream),
          v.begin<ColumnType>(),
          v.end<ColumnType>(),
          m_view.begin<ColumnType>() + count);
      count += v.size();
    }

    //If concatenated column is nullable, proceed to calculate it
    if (col->nullable()) {
      cudf::detail::concatenate_masks(views,
          (col->mutable_view()).null_mask(), stream);
    }

    return col;
  }

};

template <typename T, size_type block_size, bool Nullable>
__global__ void fused_concatenate_kernel(column_device_view const* input_views,
                                         size_type const* input_offsets,
                                         size_type num_input_views,
                                         mutable_column_device_view output_view,
                                         size_type* out_valid_count) {
  auto const output_size = output_view.size();
  auto* output_data = output_view.data<T>();

  size_type output_index = threadIdx.x + blockIdx.x * blockDim.x;
  size_type warp_valid_count = 0;

  unsigned active_mask;
  if (Nullable) {
    active_mask = __ballot_sync(0xFFFF'FFFF, output_index < output_size);
  }
  while (output_index < output_size) {

    // Lookup input index by searching for output index in offsets
    // thrust::prev isn't in CUDA 10.0, so subtracting 1 here instead
    auto const offset_it = -1 +
        thrust::upper_bound(thrust::seq, input_offsets,
                            input_offsets + num_input_views, output_index);
    size_type const partition_index = offset_it - input_offsets;

    // Copy input data to output
    auto const offset_index = output_index - *offset_it;
    auto const& input_view = input_views[partition_index];
    auto const* input_data = input_view.data<T>();
    output_data[output_index] = input_data[offset_index];

    if (Nullable) {
      bool const bit_is_set = input_view.is_valid(offset_index);
      bitmask_type const new_word = __ballot_sync(active_mask, bit_is_set);

      // First thread writes bitmask word
      if (threadIdx.x % experimental::detail::warp_size == 0) {
        output_view.null_mask()[word_index(output_index)] = new_word;
      }

      warp_valid_count += __popc(new_word);
    }

    output_index += blockDim.x * gridDim.x;
    if (Nullable) {
      active_mask = __ballot_sync(active_mask, output_index < output_size);
    }
  }

  if (Nullable) {
    using experimental::detail::single_lane_block_sum_reduce;
    auto block_valid_count = single_lane_block_sum_reduce<block_size, 0>(warp_valid_count);
    if (threadIdx.x == 0) {
      atomicAdd(out_valid_count, block_valid_count);
    }
  }
}

template <size_type block_size, bool Nullable>
__global__ void
fused_concatenate_string_offset_kernel(column_device_view const* input_views,
                                       size_type const* input_offsets,
                                       size_type const* partition_offsets,
                                       size_type const num_input_views,
                                       size_type const output_size,
                                       size_type* output_data,
                                       bitmask_type* output_mask,
                                       size_type* out_valid_count) {
  size_type output_index = threadIdx.x + blockIdx.x * blockDim.x;
  size_type warp_valid_count = 0;

  unsigned active_mask;
  if (Nullable) {
    active_mask = __ballot_sync(0xFFFF'FFFF, output_index < output_size);
  }
  while (output_index < output_size) {

    // Lookup input index by searching for output index in offsets
    // thrust::prev isn't in CUDA 10.0, so subtracting 1 here instead
    auto const offset_it = -1 +
        thrust::upper_bound(thrust::seq, input_offsets,
                            input_offsets + num_input_views, output_index);
    size_type const partition_index = offset_it - input_offsets;

    auto const offset_index = output_index - *offset_it;
    auto const& input_view = input_views[partition_index];
    constexpr auto offsets_child = strings_column_view::offsets_column_index;
    auto const* input_data = input_view.child(offsets_child).data<int32_t>();
    output_data[output_index] =
        input_data[offset_index + input_view.offset()] // handle parent offset
        - input_data[input_view.offset()] // subract first offset if non-zero
        + partition_offsets[partition_index]; // add cumulative chars offset

    if (Nullable) {
      bool const bit_is_set = input_view.is_valid(offset_index);
      bitmask_type const new_word = __ballot_sync(active_mask, bit_is_set);

      // First thread writes bitmask word
      if (threadIdx.x % experimental::detail::warp_size == 0) {
        output_mask[word_index(output_index)] = new_word;
      }

      warp_valid_count += __popc(new_word);
    }

    output_index += blockDim.x * gridDim.x;
    if (Nullable) {
      active_mask = __ballot_sync(active_mask, output_index < output_size);
    }
  }

  // Fill final offsets index with total size of char data
  if (output_index == output_size) {
    output_data[output_size] = partition_offsets[num_input_views];
  }

  if (Nullable) {
    using experimental::detail::single_lane_block_sum_reduce;
    auto block_valid_count = single_lane_block_sum_reduce<block_size, 0>(warp_valid_count);
    if (threadIdx.x == 0) {
      atomicAdd(out_valid_count, block_valid_count);
    }
  }
}

__global__ void
fused_concatenate_string_chars_kernel(column_device_view const* input_views,
                                      size_type const* partition_offsets,
                                      size_type const num_input_views,
                                      size_type const output_size,
                                      int8_t* output_data) {
  size_type output_index = threadIdx.x + blockIdx.x * blockDim.x;

  while (output_index < output_size) {

    // Lookup input index by searching for output index in offsets
    // thrust::prev isn't in CUDA 10.0, so subtracting 1 here instead
    auto const offset_it = -1 +
        thrust::upper_bound(thrust::seq, partition_offsets,
                            partition_offsets + num_input_views, output_index);
    size_type const partition_index = offset_it - partition_offsets;

    auto const offset_index = output_index - *offset_it;
    auto const& input_view = input_views[partition_index];

    constexpr auto offsets_child = strings_column_view::offsets_column_index;
    auto const* input_offsets_data = input_view.child(offsets_child).data<int32_t>();

    constexpr auto chars_child = strings_column_view::chars_column_index;
    auto const* input_chars_data = input_view.child(chars_child).data<int8_t>();

    auto const first_char = input_offsets_data[input_view.offset()];
    output_data[output_index] = input_chars_data[offset_index + first_char];

    output_index += blockDim.x * gridDim.x;
  }
}

struct fused_concatenate {
  std::vector<column_view> const& views;
  bool const has_nulls;
  rmm::mr::device_memory_resource* mr;
  cudaStream_t stream;

  template <typename T,
      std::enable_if_t<is_fixed_width<T>()>* = nullptr>
  std::unique_ptr<column> operator()() {
    using mask_policy = cudf::experimental::mask_allocation_policy;

    // Preprocess and upload inputs to device memory
    auto const device_views = create_device_views(views, stream);
    auto const& d_views = std::get<1>(device_views);
    auto const& d_offsets = std::get<2>(device_views);
    auto const output_size = std::get<3>(device_views);

    // Allocate output
    auto const policy = has_nulls ? mask_policy::ALWAYS : mask_policy::NEVER;
    auto out_col = experimental::detail::allocate_like(views.front(),
        output_size, policy, mr, stream);
    out_col->set_null_count(0); // prevent null count from being materialized
    auto out_view = out_col->mutable_view();
    auto d_out_view = mutable_column_device_view::create(out_view, stream);

    rmm::device_scalar<size_type> d_valid_count(0);

    // Launch kernel
    constexpr size_type block_size{256};
    cudf::experimental::detail::grid_1d config(output_size, block_size);
    auto const kernel = has_nulls
        ? fused_concatenate_kernel<T, block_size, true>
        : fused_concatenate_kernel<T, block_size, false>;
    kernel<<<config.num_blocks, config.num_threads_per_block, 0, stream>>>(
        d_views.data().get(),
        d_offsets.data().get(),
        static_cast<size_type>(d_views.size()),
        *d_out_view,
        d_valid_count.data());

    if (has_nulls) {
      out_col->set_null_count(output_size - d_valid_count.value(stream));
    }

    return out_col;
  }

  template <typename T,
      std::enable_if_t<std::is_same<T, cudf::dictionary32>::value>* = nullptr>
  std::unique_ptr<column> operator()() {
    CUDF_FAIL("dictionary concatenate not yet supported");
  }

  template <typename T,
      std::enable_if_t<std::is_same<T, cudf::string_view>::value>* = nullptr>
  std::unique_ptr<column> operator()() {
    using mask_policy = cudf::experimental::mask_allocation_policy;

    // Preprocess and upload inputs to device memory
    auto const device_views = create_strings_device_views(views, stream);
    auto const& d_views = std::get<1>(device_views);
    auto const& d_input_offsets = std::get<2>(device_views);
    auto const& d_partition_offsets = std::get<3>(device_views);
    auto const output_size = std::get<4>(device_views);
    auto const output_chars_size = std::get<5>(device_views);
    auto const output_offsets_size = output_size + 1;

    // Allocate child columns and null mask
    auto chars_column = make_numeric_column(data_type{INT8},
        output_chars_size, mask_state::UNALLOCATED, stream, mr);
    auto offsets_column = make_numeric_column(data_type{INT32},
        output_offsets_size, mask_state::UNALLOCATED, stream, mr);
    rmm::device_buffer null_mask;
    if (has_nulls) {
      null_mask = create_null_mask(output_size, mask_state::UNINITIALIZED, stream, mr);
    }
    size_type null_count{0};

    auto chars_view = chars_column->mutable_view();
    auto offsets_view = offsets_column->mutable_view();

    { // Launch offsets kernel
      rmm::device_scalar<size_type> d_valid_count(0);

      constexpr size_type block_size{256};
      cudf::experimental::detail::grid_1d config(output_offsets_size, block_size);
      auto const kernel = has_nulls
          ? fused_concatenate_string_offset_kernel<block_size, true>
          : fused_concatenate_string_offset_kernel<block_size, false>;
      kernel<<<config.num_blocks, config.num_threads_per_block, 0, stream>>>(
          d_views.data().get(),
          d_input_offsets.data().get(),
          d_partition_offsets.data().get(),
          static_cast<size_type>(d_views.size()),
          output_size,
          offsets_view.data<size_type>(),
          reinterpret_cast<bitmask_type*>(null_mask.data()),
          d_valid_count.data());

      if (has_nulls) {
        null_count = output_size - d_valid_count.value(stream);
      }
    }

    { // Launch chars kernel
      constexpr size_type block_size{256};
      cudf::experimental::detail::grid_1d config(output_chars_size, block_size);
      auto const kernel = fused_concatenate_string_chars_kernel;
      kernel<<<config.num_blocks, config.num_threads_per_block, 0, stream>>>(
          d_views.data().get(),
          d_partition_offsets.data().get(),
          static_cast<size_type>(d_views.size()),
          output_chars_size,
          chars_view.data<int8_t>());
    }

    return make_strings_column(output_size,
        std::move(offsets_column), std::move(chars_column),
        null_count, std::move(null_mask), stream, mr);
  }
};

// Concatenates the elements from a vector of column_views
std::unique_ptr<column>
concatenate(std::vector<column_view> const& columns_to_concat,
            rmm::mr::device_memory_resource *mr,
            cudaStream_t stream) {

  CUDF_EXPECTS(not columns_to_concat.empty(),
               "Unexpected empty list of columns to concatenate.");

  data_type const type = columns_to_concat.front().type();
  CUDF_EXPECTS(std::all_of(columns_to_concat.begin(), columns_to_concat.end(),
      [&type](auto const& c) { return c.type() == type; }),
      "Type mismatch in columns to concatenate.");

  if (std::all_of(columns_to_concat.begin(), columns_to_concat.end(),
                  [](column_view const& c) { return c.is_empty(); })) {
    return experimental::empty_like(columns_to_concat.front());
  }

  bool const has_nulls = std::any_of(
      columns_to_concat.begin(), columns_to_concat.end(),
      [](auto const& col) { return col.has_nulls(); });
  bool const fixed_width = cudf::is_fixed_width(type);
  bool const strings_col = (type.id() == type_id::STRING);

  // Select fused kernel when it can improve performance
  // TODO benchmark strings performance and update heuristic
  if (strings_col ||
      (fixed_width && (has_nulls || columns_to_concat.size() > 4))) {
    return experimental::type_dispatcher(type,
        detail::fused_concatenate{columns_to_concat, has_nulls, mr, stream});
  } else {
    return experimental::type_dispatcher(type,
        detail::for_each_concatenate{columns_to_concat, has_nulls, stream, mr});
  }
}

}  // namespace detail

rmm::device_buffer concatenate_masks(std::vector<column_view> const& views,
                                     rmm::mr::device_memory_resource* mr) {
  bool const has_nulls =
      std::any_of(views.begin(), views.end(),
                  [](const column_view col) { return col.has_nulls(); });
  if (has_nulls) {
    size_type const total_element_count = std::accumulate(
        views.begin(), views.end(), 0,
        [](auto accumulator, auto const& v) { return accumulator + v.size(); });

    rmm::device_buffer null_mask =
        create_null_mask(total_element_count, mask_state::UNINITIALIZED, 0, mr);

    detail::concatenate_masks(views,
                              static_cast<bitmask_type*>(null_mask.data()), 0);

    return null_mask;
  }
  // no nulls, so return an empty device buffer
  return rmm::device_buffer{};
}

// Concatenates the elements from a vector of column_views
std::unique_ptr<column>
concatenate(std::vector<column_view> const& columns_to_concat,
            rmm::mr::device_memory_resource *mr) {
  CUDF_FUNC_RANGE();
  return detail::concatenate(columns_to_concat, mr, 0);
}

namespace experimental {

std::unique_ptr<table>
concatenate(std::vector<table_view> const& tables_to_concat,
            rmm::mr::device_memory_resource *mr) {
  if (tables_to_concat.size() == 0) { return std::make_unique<table>(); }

  table_view const first_table = tables_to_concat.front();
  CUDF_EXPECTS(std::all_of(tables_to_concat.begin(), tables_to_concat.end(),
                           [&first_table](auto const& t) {
                             return t.num_columns() ==
                                        first_table.num_columns() &&
                                        have_same_types(first_table, t);
                           }),
               "Mismatch in table columns to concatenate.");

  std::vector<std::unique_ptr<column>> concat_columns;
  for (size_type i = 0; i < first_table.num_columns(); ++i) {
    std::vector<column_view> cols;
    for (auto &t : tables_to_concat) {
      cols.emplace_back(t.column(i));
    }
    concat_columns.emplace_back(cudf::concatenate(cols, mr));
  }
  return std::make_unique<table>(std::move(concat_columns));
}

}  // namespace experimental

}  // namespace cudf
