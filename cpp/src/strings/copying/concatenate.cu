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

#include <cudf/copying.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/column/column_device_view.cuh>
#include <cudf/strings/detail/concatenate.hpp>
#include <cudf/detail/concatenate.cuh>
#include <cudf/detail/utilities/cuda.cuh>
#include <cudf/strings/strings_column_view.hpp>
#include <strings/utilities.hpp>

#include <thrust/binary_search.h>
#include <thrust/for_each.h>
#include <thrust/transform_reduce.h>
#include <thrust/transform_scan.h>

namespace cudf
{
namespace strings
{
namespace detail
{

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

std::unique_ptr<column> fused_concatenate(
    std::vector<column_view> const& views,
    bool const has_nulls,
    rmm::mr::device_memory_resource* mr,
    cudaStream_t stream) {

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

std::unique_ptr<column> concatenate( std::vector<column_view> const& columns,
                                     rmm::mr::device_memory_resource* mr,
                                     cudaStream_t stream )
{
    // calculate the size of the output column
    size_t strings_count = thrust::transform_reduce( columns.begin(), columns.end(),
        [] (auto col) { return col.size(); }, static_cast<size_t>(0), thrust::plus<size_t>());
    CUDF_EXPECTS( strings_count < std::numeric_limits<size_type>::max(), 
        "total number of strings is too large for cudf column" );
    if( strings_count == 0 )
        return make_empty_strings_column(mr,stream);

    // build vector of column_device_views
    std::vector<std::unique_ptr<column_device_view,std::function<void(column_device_view*)> > > 
        device_cols(columns.size());
    thrust::host_vector<column_device_view> h_device_views;
    for( auto&& col : columns )
    {
        device_cols.emplace_back(column_device_view::create(col, stream));
        h_device_views.push_back(*(device_cols.back()));
    }
    rmm::device_vector<column_device_view> device_views(h_device_views);
    auto execpol = rmm::exec_policy(stream);
    auto d_views = device_views.data().get();
    // compute size of the output chars column
    size_t total_bytes = thrust::transform_reduce( execpol->on(stream),
        d_views, d_views + device_views.size(),
        [] __device__ (auto d_view) {
            if( d_view.size()==0 )
                return static_cast<size_t>(0);
            auto d_offsets = d_view.child(strings_column_view::offsets_column_index).template data<int32_t>();
            size_t size = d_offsets[d_view.size()+d_view.offset()] - d_offsets[d_view.offset()];
            return size;
        }, static_cast<size_t>(0), thrust::plus<size_t>());
    CUDF_EXPECTS( total_bytes < std::numeric_limits<size_type>::max(), "total size of strings is too large for cudf column" );

    bool const has_nulls = std::any_of(
        columns.begin(), columns.end(),
        [](auto const& col) { return col.has_nulls(); });

    // TODO refactor the column_device_view creation and offset/size
    // computation above with create_strings_device_views
    // Select fused kernel optimization where it may perform better
    bool const use_fused_kernels = has_nulls
        ? strings_count < columns.size() * 16384
        : strings_count < columns.size() * 4096;
    if (use_fused_kernels) {
      return fused_concatenate(columns, has_nulls, mr, stream);
    }

    // create chars column
    auto chars_column = make_numeric_column( data_type{INT8}, total_bytes, mask_state::UNALLOCATED, stream, mr);
    auto d_new_chars = chars_column->mutable_view().data<char>();

    // create offsets column
    auto offsets_column = make_numeric_column( data_type{INT32}, strings_count + 1, mask_state::UNALLOCATED, stream, mr);
    auto offsets_view = offsets_column->mutable_view();
    auto d_new_offsets = offsets_view.data<int32_t>();

    // copy over the data for all the columns
    ++d_new_offsets; // skip the first element which will be set to 0 after the for-loop
    int32_t offset_adjust = 0; // each section of offsets must be adjusted
    size_type null_count = 0;  // add up the null counts
    for( auto column = columns.begin(); column != columns.end(); ++column )
    {
        size_type column_size = column->size();
        if( column_size==0 ) // nothing to do
            continue; // empty column may not have children
        size_type column_offset = column->offset();
        column_view offsets_child = column->child(strings_column_view::offsets_column_index);
        column_view chars_child = column->child(strings_column_view::chars_column_index);

        // copy the offsets column
        auto d_offsets = offsets_child.data<int32_t>() + column_offset;
        int32_t bytes_offset = thrust::device_pointer_cast(d_offsets)[0];
        
        thrust::transform( rmm::exec_policy(stream)->on(stream), d_offsets + 1, d_offsets + column_size + 1, d_new_offsets,
            [offset_adjust, bytes_offset] __device__ (int32_t old_offset) {
                return old_offset - bytes_offset + offset_adjust;
            } );

        // copy the chars column data
        auto d_chars = chars_child.data<char>() + bytes_offset;
        size_type bytes = thrust::device_pointer_cast(d_offsets)[column_size] - bytes_offset;
        CUDA_TRY(cudaMemcpyAsync( d_new_chars, d_chars, bytes, cudaMemcpyDeviceToDevice, stream ));
        // get ready for the next column
        offset_adjust += bytes;
        d_new_chars += bytes;
        d_new_offsets += column_size;
        null_count += column->null_count();
    }
    CUDA_TRY(cudaMemsetAsync( offsets_view.data<int32_t>(), 0, sizeof(int32_t), stream));

    rmm::device_buffer null_mask;
    if( null_count > 0 ) {
        null_mask = create_null_mask( strings_count, mask_state::UNINITIALIZED, stream,mr );
        auto mask_data = reinterpret_cast<bitmask_type*>(null_mask.data());
        cudf::detail::concatenate_masks(columns, mask_data, stream);
    }
    offsets_column->set_null_count(0);  // reset the null counts
    chars_column->set_null_count(0);    // for children columns
    return make_strings_column(strings_count, std::move(offsets_column), std::move(chars_column),
                               null_count, std::move(null_mask), stream, mr);
}

} // namespace detail
} // namespace strings
} // namespace cudf
