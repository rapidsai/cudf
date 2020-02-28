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
#include <cudf/detail/copy.hpp>
#include <cudf/detail/utilities/cuda.cuh>
#include <cudf/strings/detail/concatenate.hpp>
#include <cudf/utilities/nvtx_utils.hpp>

#include <thrust/binary_search.h>
#include <thrust/transform_scan.h>

#include <algorithm>
#include <numeric>

namespace cudf {
namespace detail {

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
    size_type source_view_index = thrust::upper_bound(thrust::seq,
        output_offsets, output_offsets + number_of_views,
        mask_index) - output_offsets - 1;
    bool bit_is_set = 1;
    if (source_view_index < number_of_views) {
      size_type column_element_index = mask_index - output_offsets[source_view_index];
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

  using CDViewPtr =
    decltype(column_device_view::create(std::declval<column_view>(), std::declval<cudaStream_t>()));
  std::vector<CDViewPtr> cols;
  thrust::host_vector<column_device_view> device_views;

  thrust::host_vector<size_type> view_offsets(1, 0);
  for (auto &v : views) {
    cols.emplace_back(column_device_view::create(v, stream));
    device_views.push_back(*(cols.back()));
    view_offsets.push_back(v.size());
  }
  thrust::inclusive_scan(thrust::host,
      view_offsets.begin(), view_offsets.end(),
      view_offsets.begin());

  rmm::device_vector<column_device_view> d_views{device_views};
  rmm::device_vector<size_type> d_offsets{view_offsets};

  auto number_of_mask_bits = view_offsets.back();

  concatenate_masks(d_views, d_offsets, dest_mask, number_of_mask_bits, stream);
}

}  // namespace detail

rmm::device_buffer concatenate_masks(std::vector<column_view> const &views,
                                     rmm::mr::device_memory_resource *mr,
                                     cudaStream_t stream) {
  rmm::device_buffer null_mask{};
  bool has_nulls = std::any_of(views.begin(), views.end(),
                     [](const column_view col) { return col.has_nulls(); });
  if (has_nulls) {
   size_type total_element_count =
     std::accumulate(views.begin(), views.end(), 0,
         [](auto accumulator, auto const& v) { return accumulator + v.size(); });
    null_mask = create_null_mask(total_element_count, mask_state::UNINITIALIZED, stream, mr);

    detail::concatenate_masks(
        views, static_cast<bitmask_type *>(null_mask.data()), stream);
  }

  return null_mask;
}


struct create_column_from_view_vector {
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

   auto type = views.front().type();
   size_type total_element_count =
     std::accumulate(views.begin(), views.end(), 0,
         [](auto accumulator, auto const& v) { return accumulator + v.size(); });

   bool has_nulls = std::any_of(views.begin(), views.end(),
                      [](const column_view col) { return col.has_nulls(); });
   using mask_policy = cudf::experimental::mask_allocation_policy;

   mask_policy policy{mask_policy::NEVER};
   if (has_nulls) { policy = mask_policy::ALWAYS; }

   auto col = cudf::experimental::allocate_like(views.front(),
       total_element_count, policy, mr);

   auto m_view = col->mutable_view();
   auto count = 0;
   // TODO replace loop with a single kernel https://github.com/rapidsai/cudf/issues/2881
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

template <typename T>
struct partition_map_fn {
  size_type const* offsets_ptr;
  size_type const* partition_ptr;
  T const* const* data_ptr;

  T __device__ operator()(size_type i) {
    auto const partition = partition_ptr[i];
    auto const offset = offsets_ptr[partition];
    return data_ptr[partition][i - offset];
  }
};

template <typename T>
struct binary_search_fn {
  size_type const* offsets_ptr;
  size_t const input_size;
  T const* const* data_ptr;

  T __device__ operator()(size_type i) {
    // Look up the current index in the offsets table to find the partition
    // Select the offset before the upper bound
    auto const offset_it = thrust::upper_bound(thrust::seq,
        offsets_ptr, offsets_ptr + input_size, i) - 1;
    auto const partition = offset_it - offsets_ptr;
    auto const offset = *offset_it;
    return data_ptr[partition][i - offset];
  }
};

auto compute_partition_map(rmm::device_vector<size_type> const& d_offsets,
    size_t output_size, cudaStream_t stream) {

  auto d_partition_map = rmm::device_vector<size_type>(output_size);

  // Scatter 1s at the start of each partition, then scan to fill the map
  auto const const_it = thrust::make_constant_iterator(size_type{1});
  thrust::scatter(rmm::exec_policy(stream)->on(stream),
      const_it, const_it + (d_offsets.size() - 2), // skip first and last offset
      std::next(d_offsets.cbegin()), // skip first offset, leaving it set to 0
      d_partition_map.begin());

  thrust::inclusive_scan(rmm::exec_policy(stream)->on(stream),
      d_partition_map.cbegin(), d_partition_map.cend(),
      d_partition_map.begin());

  return d_partition_map;
}

// Allow strategy switching at runtime for easier benchmarking
// TODO remove when done
static concatenate_mode current_mode = concatenate_mode::UNOPTIMIZED;
void temp_set_concatenate_mode(concatenate_mode mode) {
  current_mode = mode;
}

struct optimized_concatenate {
  template <typename T,
      std::enable_if_t<is_fixed_width<T>()>* = nullptr>
  std::unique_ptr<column> operator()(
      std::vector<column_view> const& views,
      rmm::mr::device_memory_resource* mr,
      cudaStream_t stream) {
    using mask_policy = cudf::experimental::mask_allocation_policy;

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

    // Transform views to array of data pointers
    auto data_ptrs = thrust::host_vector<T const*>(views.size());
    std::transform(views.begin(), views.end(), data_ptrs.begin(),
        [](column_view const& col) {
          return col.data<T>();
        });
    auto const d_data_ptrs = rmm::device_vector<T const*>{data_ptrs};

    // Allocate output column, with null mask if any columns have nulls
    bool const has_nulls = std::any_of(views.begin(), views.end(),
        [](auto const& col) { return col.has_nulls(); });
    auto const policy = has_nulls ? mask_policy::ALWAYS : mask_policy::NEVER;
    auto out_col = experimental::detail::allocate_like(views.front(),
        output_size, policy, mr, stream);
    auto out_view = out_col->mutable_view();

    // Initialize each output row from its corresponding input view
    // NOTE device lambdas were giving weird errors, so use functor instead
    auto const* offsets_ptr = d_offsets.data().get();
    T const* const* data_ptr = d_data_ptrs.data().get();
    if (current_mode == concatenate_mode::PARTITION_MAP) {
      auto const d_partition_map = compute_partition_map(d_offsets, output_size, stream);
      auto const* partition_ptr = d_partition_map.data().get();
      thrust::tabulate(rmm::exec_policy(stream)->on(stream),
          out_view.begin<T>(), out_view.end<T>(),
          partition_map_fn<T>{offsets_ptr, partition_ptr, data_ptr});
    } else {
      thrust::tabulate(rmm::exec_policy(stream)->on(stream),
          out_view.begin<T>(), out_view.end<T>(),
          binary_search_fn<T>{offsets_ptr, views.size(), data_ptr});
    }

    // If concatenated column is nullable, proceed to calculate it
    // TODO try to fuse this with the data kernel so they share binary search
    if (has_nulls) {
      detail::concatenate_masks(d_views, d_offsets, out_view.null_mask(),
          output_size, stream);
    }

    return out_col;
  }

  template <typename ColumnType,
      std::enable_if_t<not is_fixed_width<ColumnType>()>* = nullptr>
  std::unique_ptr<column> operator()(
      std::vector<column_view> const& views,
      rmm::mr::device_memory_resource* mr,
      cudaStream_t stream) {
    CUDF_FAIL("non-fixed-width types not yet supported");
  }
};

struct nvtx_raii {
  nvtx_raii(char const* name, nvtx::color color) { nvtx::range_push(name, color); }
  ~nvtx_raii() { nvtx::range_pop(); }
};

// Concatenates the elements from a vector of column_views
std::unique_ptr<column>
concatenate(std::vector<column_view> const& columns_to_concat,
            rmm::mr::device_memory_resource *mr, cudaStream_t stream) {
//#if defined(BUILD_BENCHMARKS)
// TODO this doesn't seem to work, so remove after testing
// This shows additional information for profiling in NVTX ranges,
// but at the expense of some extra computation
#if true // for testing, print [num_cols][rows_per_col]
  auto const num_cols = columns_to_concat.size();
  // This should be thrust::transform_reduce, but getting
  // error related to https://github.com/rapidsai/rmm/pull/312
  auto col_sizes = std::vector<size_type>(num_cols);
  thrust::transform(
      columns_to_concat.cbegin(), columns_to_concat.cend(),
      col_sizes.begin(),
      [] __host__ (auto const& col) {
        return col.size();
      });
  auto const rows_per_col = std::accumulate(
      col_sizes.begin(), col_sizes.end(), size_type{0});
  auto const message = std::string("cudf::concatenate[")
      + std::to_string(num_cols) + "]["
      + std::to_string(rows_per_col / num_cols) + "]";
  nvtx_raii range(message.c_str(), nvtx::color::DARK_GREEN);
#else
  nvtx_raii range("cudf::concatenate", nvtx::color::DARK_GREEN);
#endif

  if (columns_to_concat.empty()) { return std::make_unique<column>(); }

  data_type type = columns_to_concat.front().type();
  CUDF_EXPECTS(std::all_of(columns_to_concat.begin(), columns_to_concat.end(),
        [type](auto const& c) { return c.type() == type; }),
      "Type mismatch in columns to concatenate.");

  switch (current_mode) {
    case concatenate_mode::UNOPTIMIZED:
      return experimental::type_dispatcher(type,
          create_column_from_view_vector{columns_to_concat, stream, mr});
    case concatenate_mode::PARTITION_MAP:
    case concatenate_mode::BINARY_SEARCH:
      return experimental::type_dispatcher(type,
          optimized_concatenate{}, columns_to_concat, mr, stream);
    default:
      CUDF_FAIL("Invalid concatenate mode");
  }
}

namespace experimental {

std::unique_ptr<table>
concatenate(std::vector<table_view> const& tables_to_concat,
            rmm::mr::device_memory_resource *mr, cudaStream_t stream) {
  if (tables_to_concat.size() == 0) { return std::make_unique<table>(); }

  size_type number_of_cols = tables_to_concat.front().num_columns();
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
