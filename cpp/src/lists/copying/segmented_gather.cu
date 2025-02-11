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
#include <cudf/detail/copy.hpp>
#include <cudf/detail/copy_range.cuh>
#include <cudf/detail/gather.cuh>
#include <cudf/detail/indexalator.cuh>
#include <cudf/detail/iterator.cuh>
#include <cudf/detail/null_mask.hpp>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/lists/detail/gather.cuh>
#include <cudf/lists/filling.hpp>
#include <cudf/lists/gather.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/memory_resource.hpp>
#include <cudf/detail/sizes_to_offsets_iterator.cuh>

#include <rmm/cuda_stream_view.hpp>

#include <cuda/functional>
#include <thrust/binary_search.h>
#include <thrust/execution_policy.h>

namespace cudf {
namespace lists {
namespace detail {

std::unique_ptr<column> segmented_gather(lists_column_view const& value_column,
                                         lists_column_view const& gather_map,
                                         out_of_bounds_policy bounds_policy,
                                         rmm::cuda_stream_view stream,
                                         rmm::device_async_resource_ref mr)
{
  CUDF_EXPECTS(is_index_type(gather_map.child().type()),
               "Gather map should be list column of index type");
  CUDF_EXPECTS(!gather_map.has_nulls(), "Gather map contains nulls", std::invalid_argument);
  CUDF_EXPECTS(value_column.size() == gather_map.size(),
               "Gather map and list column should be same size");

  auto const gather_map_sliced_child = gather_map.get_sliced_child(stream);
  auto const gather_map_size         = gather_map_sliced_child.size();
  auto const gather_index_begin      = gather_map.offsets_begin() + 1;
  auto const gather_index_end        = gather_map.offsets_end();
  auto const value_offsets           = value_column.offsets_begin();
  auto const value_device_view       = column_device_view::create(value_column.parent(), stream);
  auto const map_begin =
    cudf::detail::indexalator_factory::make_input_iterator(gather_map_sliced_child);
  auto const out_of_bounds = [] __device__(auto const index, auto const list_size) {
    return index >= list_size || (index < 0 && -index > list_size);
  };

  // Calculate Flattened gather indices  (value_offset[row]+sub_index
  auto transformer =
    cuda::proclaim_return_type<size_type>([values_lists_view = *value_device_view,
                                           value_offsets,
                                           map_begin,
                                           gather_index_begin,
                                           gather_index_end,
                                           bounds_policy,
                                           out_of_bounds] __device__(size_type index) -> size_type {
      // Get each row's offset. (Each row is a list).
      auto offset_idx =
        thrust::upper_bound(
          thrust::seq, gather_index_begin, gather_index_end, gather_index_begin[-1] + index) -
        gather_index_begin;
      // Get each sub_index in list in each row of gather_map.
      auto sub_index    = map_begin[index];
      auto list_is_null = values_lists_view.is_null(offset_idx);
      auto list_size =
        list_is_null ? 0 : (value_offsets[offset_idx + 1] - value_offsets[offset_idx]);
      auto wrapped_sub_index  = sub_index < 0 ? sub_index + list_size : sub_index;
      auto constexpr null_idx = cuda::std::numeric_limits<cudf::size_type>::max();
      // Add sub_index to value_column offsets, to get gather indices of child of value_column
      return (bounds_policy == out_of_bounds_policy::NULLIFY && out_of_bounds(sub_index, list_size))
               ? null_idx
               : value_offsets[offset_idx] + wrapped_sub_index - value_offsets[0];
    });
  auto child_gather_index_begin = cudf::detail::make_counting_transform_iterator(0, transformer);

  // Call gather on child of value_column
  auto child_table = cudf::detail::gather(table_view({value_column.get_sliced_child(stream)}),
                                          child_gather_index_begin,
                                          child_gather_index_begin + gather_map_size,
                                          bounds_policy,
                                          stream,
                                          mr);
  auto child       = std::move(child_table->release().front());

  // Create list offsets from gather_map.
  auto output_offset = cudf::detail::allocate_like(
    gather_map.offsets(), gather_map.size() + 1, mask_allocation_policy::RETAIN, stream, mr);
  auto output_offset_view = output_offset->mutable_view();
  cudf::detail::copy_range_in_place(gather_map.offsets(),
                                    output_offset_view,
                                    gather_map.offset(),
                                    gather_map.offset() + output_offset_view.size(),
                                    0,
                                    stream);
  // Assemble list column & return
  auto null_mask       = cudf::detail::copy_bitmask(value_column.parent(), stream, mr);
  size_type null_count = value_column.null_count();
  return make_lists_column(gather_map.size(),
                           std::move(output_offset),
                           std::move(child),
                           null_count,
                           std::move(null_mask),
                           stream,
                           mr);
}

void assert_start_is_not_zero(int start)
{
  CUDF_EXPECTS(start != 0, "Invalid start value: start must not be 0");
}

void assert_start_is_not_zero(cudf::column_view const& start, rmm::cuda_stream_view stream)
{
  bool start_valid = thrust::all_of(rmm::exec_policy(stream),
                                    start.begin<int32_t>(),
                                    start.end<int32_t>(),
                                    [] __device__(int32_t x) { return x != 0; });
  CUDF_EXPECTS(start_valid, "Invalid start value: start must not be 0");
}

void assert_length_is_not_negative(int length)
{
  CUDF_EXPECTS(length >= 0, "Invalid length value: length must be >= 0");
}

void assert_length_is_not_negative(cudf::column_view const& length, rmm::cuda_stream_view stream)
{
  bool length_valid = thrust::all_of(rmm::exec_policy(stream),
                                     length.begin<int32_t>(),
                                     length.end<int32_t>(),
                                     [] __device__(int32_t x) { return x >= 0; });
  CUDF_EXPECTS(length_valid, "Invalid length value: length must be >= 0");
}

struct int_iterator_from_scalar {
  int32_t value;
  int_iterator_from_scalar(int32_t v) : value(v) {}
  __device__ int32_t operator()(cudf::thread_index_type const index) const
  {
    // ignore index, always return the same value
    return value;
  }
};

struct int_iterator_from_pointer {
  int32_t const* pointer;
  int_iterator_from_pointer(int32_t const* p) : pointer(p) {}
  __device__ int32_t operator()(cudf::thread_index_type const index) const { return pointer[index]; }
};

template <typename INT_ITERATOR>
CUDF_KERNEL void compute_sizes_kernel(cudf::size_type const* offsets_of_input_lists,
                                      cudf::size_type const num_rows_of_input,
                                      INT_ITERATOR const start_iterator,
                                      INT_ITERATOR const length_iterator,
                                      cudf::size_type* d_index_to_copy, // indices for each sub list
                                      cudf::size_type* d_sizes_to_copy)
{
  auto const tid = cudf::detail::grid_1d::global_thread_id();
  if (tid >= num_rows_of_input) { return; }

  auto const begin          = offsets_of_input_lists[tid];
  auto const end            = offsets_of_input_lists[tid + 1];
  auto const length_of_list = end - begin;
  auto const start          = start_iterator(tid);
  auto const length         = length_iterator(tid);
  if (start < 0) {
    auto const real_start = length_of_list + start;
    if (real_start < 0) {
      d_sizes_to_copy[tid] = 0;
    } else {
      d_index_to_copy[tid] = real_start;
      auto s = length_of_list - real_start;
      s = s < 0 ? 0 : s;
      d_sizes_to_copy[tid] = cuda::std::min(s, length);
    }
  } else {
    auto const real_start = start - 1;
    if (real_start >= length_of_list) {
      d_sizes_to_copy[tid] = 0;
    } else {
      d_index_to_copy[tid] = real_start;
      auto s = length_of_list - real_start;
      s = s < 0 ? 0 : s;
      d_sizes_to_copy[tid] = cuda::std::min(s, length);
    }
  }
}

// option 1
std::unique_ptr<column> segmented_gather(lists_column_view const& source_column,
                                         column_view const& start,
                                         column_view const& length,
                                         rmm::cuda_stream_view stream,
                                         rmm::device_async_resource_ref mr)
{
  auto d_child = cudf::column_device_view::create(source_column.child(), stream);

  // compute the index and length of each list to copy
  auto const index_to_copy = make_numeric_column(data_type{cudf::type_id::INT32},
                                                 source_column.size(),
                                                 mask_state::UNALLOCATED,
                                                 stream,
                                                 rmm::mr::get_current_device_resource());
  auto const sizes_to_copy = make_numeric_column(data_type{cudf::type_id::INT32},
                                                 source_column.size(),
                                                 mask_state::UNALLOCATED,
                                                 stream,
                                                 rmm::mr::get_current_device_resource());
  // TODO set block size
  compute_sizes_kernel<<<1, 1, 0, stream.value()>>>(source_column.offsets_begin(),
                       source_column.size(),
                       int_iterator_from_pointer(start.data<int32_t>()),
                       int_iterator_from_pointer(length.data<int32_t>()),
                       index_to_copy->mutable_view().data<int32_t>(),
                       sizes_to_copy->mutable_view().data<int32_t>());

  // generate gather map
  auto const gather_map =
    cudf::lists::sequences(index_to_copy->view(), sizes_to_copy->view(), stream, mr);
  cudf::lists_column_view const gather_map_view{*gather_map};

  // gather
  return segmented_gather(
    source_column, gather_map_view, cudf::out_of_bounds_policy::DONT_CHECK, stream, mr);
}

template <typename INT_ITERATOR>
CUDF_KERNEL void compute_sizes_kernel_2(cudf::size_type const* offsets_of_input_lists,
                                      cudf::size_type const num_rows_of_input,
                                      INT_ITERATOR const start_iterator,
                                      INT_ITERATOR const length_iterator,
                                      cudf::size_type* d_index_to_copy,  // indices for the whole child list
                                      cudf::size_type* d_sizes_to_copy)
{
  auto const tid = cudf::detail::grid_1d::global_thread_id();
  if (tid >= num_rows_of_input) { return; }

  auto const begin          = offsets_of_input_lists[tid];
  auto const end            = offsets_of_input_lists[tid + 1];
  auto const length_of_list = end - begin;
  auto const start          = start_iterator(tid);
  auto const length         = length_iterator(tid);
  if (start < 0) {
    auto const real_start = length_of_list + start;
    if (real_start < 0) {
      d_sizes_to_copy[tid] = 0;
    } else {
      d_index_to_copy[tid] = real_start + begin;
      auto s = length_of_list - real_start;
      s = s < 0 ? 0 : s;
      d_sizes_to_copy[tid] = cuda::std::min(s, length);
    }
  } else {
    auto const real_start = start - 1;
    if (real_start >= length_of_list) {
      d_sizes_to_copy[tid] = 0;
    } else {
      d_index_to_copy[tid] = real_start + begin;
      auto s = length_of_list - real_start;
      s = s < 0 ? 0 : s;
      d_sizes_to_copy[tid] = cuda::std::min(s, length);
    }
  }
}

CUDF_KERNEL void compute_gather_map(cudf::size_type const num_rows_of_input,
                                    cudf::size_type const* d_index_to_copy,  // indices for the whole child list
                                    cudf::size_type const* d_sizes_to_copy,
                                    cudf::size_type const* offsetalator, // 累计size
                                    cudf::size_type* d_gather_map)
{
  auto const tid = cudf::detail::grid_1d::global_thread_id();
  if (tid >= num_rows_of_input) { return; }

  auto const output = offsetalator[tid];
  auto const begin_index = d_index_to_copy[tid];
  auto const rows = d_sizes_to_copy[tid];
  for (auto i = 0; i < rows; i++)
  {
    d_gather_map[output + i] = begin_index + i;
  }
}

// option 2
std::unique_ptr<column> segmented_gather_2(lists_column_view const& source_column,
                                         column_view const& start,
                                         column_view const& length,
                                         rmm::cuda_stream_view stream,
                                         rmm::device_async_resource_ref mr)
{
  auto d_child = cudf::column_device_view::create(source_column.child(), stream);

  // compute the index and length of each list to copy
  auto const index_to_copy = make_numeric_column(data_type{cudf::type_id::INT32},
                                                 source_column.size(),
                                                 mask_state::UNALLOCATED,
                                                 stream,
                                                 rmm::mr::get_current_device_resource());
  auto const sizes_to_copy = make_numeric_column(data_type{cudf::type_id::INT32},
                                                 source_column.size(),
                                                 mask_state::UNALLOCATED,
                                                 stream,
                                                 rmm::mr::get_current_device_resource());
  compute_sizes_kernel_2<<<1, 1, 0, stream.value()>>>(source_column.offsets_begin(),
                       source_column.size(),
                       int_iterator_from_pointer(start.data<int32_t>()),
                       int_iterator_from_pointer(length.data<int32_t>()),
                       index_to_copy->mutable_view().data<int32_t>(),
                       sizes_to_copy->mutable_view().data<int32_t>());


  // make offsetalator
  auto [offsetalator, num_total_elements] = cudf::detail::make_offsets_child_column(
    sizes_to_copy->view().begin<int>(), 
    sizes_to_copy->view().end<int>(), 
    stream,
    mr);
  // generate gather map 
  rmm::device_uvector<int> d_map(num_total_elements, stream);
  compute_gather_map<<<1, 1, 0, stream.value()>>>(
    source_column.size(), 
    index_to_copy->view().begin<int>(), 
    sizes_to_copy->view().begin<int>(), 
    offsetalator->view().begin<int>(),
    d_map.data());

  // gather
  // Call gather on child of value_column
  auto child_table = cudf::detail::gather(table_view({value_column.get_sliced_child(stream)}),
                                          gather_map->view().begin<int>(),
                                          gather_map->view().end<int>(),
                                          bounds_policy,
                                          stream,
                                          mr);

                        
  auto child       = std::move(child_table->release().front());

  // Create list offsets from gather_map.
  auto output_offset = cudf::detail::allocate_like(
    gather_map.offsets(), gather_map.size() + 1, mask_allocation_policy::RETAIN, stream, mr);
  auto output_offset_view = output_offset->mutable_view();
  cudf::detail::copy_range_in_place(gather_map.offsets(),
                                    output_offset_view,
                                    gather_map.offset(),
                                    gather_map.offset() + output_offset_view.size(),
                                    0,
                                    stream);
  // Assemble list column & return
  auto null_mask       = cudf::detail::copy_bitmask(value_column.parent(), stream, mr);
  size_type null_count = value_column.null_count();
  return make_lists_column(gather_map.size(),
                           std::move(output_offset),
                           std::move(child),
                           null_count,
                           std::move(null_mask),
                           stream,
                           mr);
}

std::unique_ptr<column> segmented_gather(lists_column_view const& source_column,
                                         size_type const start,
                                         size_type const length,
                                         rmm::cuda_stream_view stream,
                                         rmm::device_async_resource_ref mr)
{
  return nullptr;
}

std::unique_ptr<column> segmented_gather(lists_column_view const& source_column,
                                         column_view const& start,
                                         size_type const length,
                                         rmm::cuda_stream_view stream,
                                         rmm::device_async_resource_ref mr)
{
  return nullptr;
}

std::unique_ptr<column> segmented_gather(lists_column_view const& source_column,
                                         size_type const start,
                                         column_view const& length,
                                         rmm::cuda_stream_view stream,
                                         rmm::device_async_resource_ref mr)
{
  return nullptr;
}

}  // namespace detail

std::unique_ptr<column> segmented_gather(lists_column_view const& source_column,
                                         lists_column_view const& gather_map_list,
                                         out_of_bounds_policy bounds_policy,
                                         rmm::cuda_stream_view stream,
                                         rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  return detail::segmented_gather(source_column, gather_map_list, bounds_policy, stream, mr);
}

std::unique_ptr<column> segmented_gather(lists_column_view const& source_column,
                                         column_view const& start,
                                         column_view const& length,
                                         rmm::cuda_stream_view stream,
                                         rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  detail::assert_start_is_not_zero(start, stream);
  detail::assert_length_is_not_negative(length, stream);
  return detail::segmented_gather(source_column, start, length, stream, mr);
}

std::unique_ptr<column> segmented_gather(lists_column_view const& source_column,
                                         size_type const start,
                                         size_type const length,
                                         rmm::cuda_stream_view stream,
                                         rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  detail::assert_start_is_not_zero(start);
  detail::assert_length_is_not_negative(length);
  return detail::segmented_gather(source_column, start, length, stream, mr);
}

std::unique_ptr<column> segmented_gather(lists_column_view const& source_column,
                                         column_view const& start,
                                         size_type const length,
                                         rmm::cuda_stream_view stream,
                                         rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  detail::assert_start_is_not_zero(start, stream);
  detail::assert_length_is_not_negative(length);
  return detail::segmented_gather(source_column, start, length, stream, mr);
}

std::unique_ptr<column> segmented_gather(lists_column_view const& source_column,
                                         size_type const start,
                                         column_view const& length,
                                         rmm::cuda_stream_view stream,
                                         rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  detail::assert_start_is_not_zero(start);
  detail::assert_length_is_not_negative(length, stream);
  return detail::segmented_gather(source_column, start, length, stream, mr);
}

}  // namespace lists
}  // namespace cudf
