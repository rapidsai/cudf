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
#include <cudf/table/table.hpp>
#include <cudf/table/table_device_view.cuh>
#include <cudf/detail/copy.hpp>
#include <cudf/detail/utilities/cuda.cuh>
#include <cudf/strings/detail/concatenate.hpp>
#include <cudf/utilities/nvtx_utils.hpp>

#include <thrust/binary_search.h>
#include <thrust/transform_scan.h>

#include <algorithm>
#include <numeric>

namespace cudf {

// Allow strategy switching at runtime for easier benchmarking
// TODO remove when done
static concatenate_mode current_mode = concatenate_mode::UNOPTIMIZED;
void temp_set_concatenate_mode(concatenate_mode mode) {
  current_mode = mode;
}

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

    bool const has_nulls = std::any_of(views.begin(), views.end(),
                        [](const column_view col) { return col.has_nulls(); });
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

template <typename T, bool Nullable>
__global__
void
fused_concatenate_kernel(
    column_device_view const* input_views,
    size_type const* input_offsets,
    size_type input_size,
    mutable_column_device_view output_view) {

  auto const output_size = output_view.size();
  auto* output_data = output_view.data<T>();

  size_type output_index = threadIdx.x + blockIdx.x * blockDim.x;
  unsigned active_mask;
  if (Nullable) {
    active_mask = __ballot_sync(0xFFFF'FFFF, output_index < output_size);
  }
  while (output_index < output_size) {

    // Lookup input index by searching for output index in offsets
    auto const offset_it = thrust::upper_bound(thrust::seq,
        input_offsets, input_offsets + input_size, output_index) - 1;
    size_type const input_index = offset_it - input_offsets;

    // Copy input data to output and read bitmask
    bool bit_is_set = false;
    if (input_index < input_size) {
      auto const offset_index = output_index - *offset_it;
      auto const& input_view = input_views[input_index];
      auto const* input_data = input_view.data<T>();
      output_data[output_index] = input_data[offset_index];
      if (Nullable) {
        bit_is_set = input_view.is_valid(offset_index);
      }
    }

    if (Nullable) {
      // TODO count set bits

      // First thread writes bitmask word
      bitmask_type const new_word = __ballot_sync(active_mask, bit_is_set);
      if (threadIdx.x % experimental::detail::warp_size == 0) {
        output_view.null_mask()[word_index(output_index)] = new_word;
      }
    }

    output_index += blockDim.x * gridDim.x;
    if (Nullable) {
      active_mask = __ballot_sync(active_mask, output_index < output_size);
    }
  }
}

struct fused_concatenate {
  template <typename T,
      std::enable_if_t<is_fixed_width<T>()>* = nullptr>
  std::unique_ptr<column> operator()(
      std::vector<column_view> const& views,
      rmm::mr::device_memory_resource* mr,
      cudaStream_t stream) {
    using mask_policy = cudf::experimental::mask_allocation_policy;

    // Preprocess and upload inputs to device memory
    auto const device_views = create_device_views(views, stream);
    auto const& d_views = std::get<1>(device_views);
    auto const& d_offsets = std::get<2>(device_views);
    auto const output_size = std::get<3>(device_views);

    bool const has_nulls = std::any_of(
      views.begin(), views.end(),
      [](auto const& col) { return col.has_nulls(); });

    // Allocate output
    auto const policy = has_nulls ? mask_policy::ALWAYS : mask_policy::NEVER;
    auto out_col = experimental::detail::allocate_like(views.front(),
        output_size, policy, mr, stream);
    out_col->set_null_count(0); // prevent null count from being materialized
    auto out_view = out_col->mutable_view();
    auto d_out_view = mutable_column_device_view::create(out_view, stream);

    // Launch kernel
    constexpr size_type block_size{256};
    cudf::experimental::detail::grid_1d config(output_size, block_size);
    auto const kernel = has_nulls
        ? fused_concatenate_kernel<T, true>
        : fused_concatenate_kernel<T, false>;
    kernel<<<config.num_blocks, config.num_threads_per_block, 0, stream>>>(
        d_views.data().get(),
        d_offsets.data().get(),
        static_cast<size_type>(d_views.size()),
        *d_out_view);

    // TODO compute null count inside the kernel and set it here

    return out_col;
  }

  template <typename T,
      std::enable_if_t<std::is_same<T, cudf::dictionary32>::value>* = nullptr>
  std::unique_ptr<column> operator()(
      std::vector<column_view> const& views,
      rmm::mr::device_memory_resource* mr,
      cudaStream_t stream) {
    CUDF_FAIL("dictionary concatenate not yet supported");
  }

  template <typename T,
      std::enable_if_t<std::is_same<T, cudf::string_view>::value>* = nullptr>
  std::unique_ptr<column> operator()(
      std::vector<column_view> const& views,
      rmm::mr::device_memory_resource* mr,
      cudaStream_t stream) {
    CUDF_FAIL("strings concatenate not yet supported");
  }
};

}  // namespace detail

rmm::device_buffer concatenate_masks(std::vector<column_view> const &views,
                                     rmm::mr::device_memory_resource *mr,
                                     cudaStream_t stream) {
  rmm::device_buffer null_mask{};
  bool const has_nulls = std::any_of(views.begin(), views.end(),
                     [](const column_view col) { return col.has_nulls(); });
  if (has_nulls) {
   size_type const total_element_count =
     std::accumulate(views.begin(), views.end(), 0,
         [](auto accumulator, auto const& v) { return accumulator + v.size(); });
    null_mask = create_null_mask(total_element_count, mask_state::UNINITIALIZED, stream, mr);

    detail::concatenate_masks(
        views, static_cast<bitmask_type *>(null_mask.data()), stream);
  }

  return null_mask;
}

// Concatenates the elements from a vector of column_views
std::unique_ptr<column>
concatenate(std::vector<column_view> const& columns_to_concat,
            rmm::mr::device_memory_resource *mr, cudaStream_t stream) {
  nvtx::raii_range range("cudf::concatenate", nvtx::color::DARK_GREEN);

  if (columns_to_concat.empty()) { return std::make_unique<column>(); }

  data_type const type = columns_to_concat.front().type();
  CUDF_EXPECTS(std::all_of(columns_to_concat.begin(), columns_to_concat.end(),
      [type](auto const& c) { return c.type() == type; }),
      "Type mismatch in columns to concatenate.");

  // TODO dispatch to fused kernel if num inputs <= 4?

  switch (current_mode) {
    case concatenate_mode::UNOPTIMIZED:
      return experimental::type_dispatcher(type,
          detail::for_each_concatenate{columns_to_concat, stream, mr});
    case concatenate_mode::FUSED_KERNEL:
      return experimental::type_dispatcher(type,
          detail::fused_concatenate{}, columns_to_concat, mr, stream);
    default:
      CUDF_FAIL("Invalid concatenate mode");
  }
}

namespace experimental {

std::unique_ptr<table>
concatenate(std::vector<table_view> const& tables_to_concat,
            rmm::mr::device_memory_resource *mr, cudaStream_t stream) {
  if (tables_to_concat.size() == 0) { return std::make_unique<table>(); }

  size_type const number_of_cols = tables_to_concat.front().num_columns();
  CUDF_EXPECTS(std::all_of(tables_to_concat.begin(), tables_to_concat.end(),
        [number_of_cols](auto const& t) { return t.num_columns() == number_of_cols; }),
      "Mismatch in table number of columns to concatenate.");

  std::vector<std::unique_ptr<column>> concat_columns;
  for (size_type i = 0; i < number_of_cols; ++i) {
    std::vector<column_view> cols;
    for (auto &t : tables_to_concat) {
      cols.emplace_back(t.column(i));
    }
    concat_columns.emplace_back(concatenate(cols, mr, stream));
  }
  return std::make_unique<table>(std::move(concat_columns));
}

}  // namespace experimental

}  // namespace cudf
