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
#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_view.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/error.hpp>
#include <numeric>

#include <rmm/thrust_rmm_allocator.h>

namespace cudf {
// Trivially copy all members but the children
column_device_view::column_device_view(column_view source)
  : detail::column_device_view_base{source.type(),
                                    source.size(),
                                    source.head(),
                                    source.null_mask(),
                                    source.offset()},
    _num_children{source.num_children()}
{
}

// Free device memory allocated for children
void column_device_view::destroy() { delete this; }

namespace {
/**
 * @brief Helper function for use by column_device_view and mutable_column_device_view constructors
 * to build device_views from views.
 *
 * It is used to build the array of child columns in device memory. Since child columns can
 * also have child columns, this uses recursion to build up the flat device buffer to contain
 * all the children and set the member pointers appropriately.
 *
 * This is accomplished by laying out all the children and grand-children into a flat host
 * buffer first but also keep a running device pointer to but used when setting the
 * d_children array result.
 *
 * This function is provided both the host pointer in which to insert its children (and
 * by recursion its grand-children) and the device pointer to be used when calculating
 * ultimate device pointer for the d_children member.
 *
 * @tparam ColumnView is either column_view or mutable_column_view
 * @tparam ColumnDeviceView is either column_device_view or mutable_column_device_view
 *
 * @param source The column view to make into a device view
 * @param h_ptr The host memory where to place any child data
 * @param d_ptr The device pointer for calculating the d_children member of any child data
 * @return The device pointer to be used for the d_children member of the given column
 */
template <typename ColumnView, typename ColumnDeviceView>
ColumnDeviceView* child_columns_to_device_array(ColumnView const& source, void* h_ptr, void* d_ptr)
{
  ColumnDeviceView* d_children = nullptr;
  size_type num_children       = source.num_children();
  if (num_children > 0) {
    // The beginning of the memory must be the fixed-sized ColumnDeviceView
    // struct objects in order for d_children to be used as an array.
    auto h_column = reinterpret_cast<ColumnDeviceView*>(h_ptr);
    auto d_column = reinterpret_cast<ColumnDeviceView*>(d_ptr);
    // Any child data is assigned past the end of this array: h_end and d_end.
    auto h_end = reinterpret_cast<int8_t*>(h_column + num_children);
    auto d_end = reinterpret_cast<int8_t*>(d_column + num_children);
    d_children = d_column;  // set children pointer for return
    for (size_type idx = 0; idx < num_children; ++idx) {
      // inplace-new each child into host memory
      auto child = source.child(idx);
      new (h_column) ColumnDeviceView(child, h_end, d_end);
      h_column++;  // advance to next child
      // update the pointers for holding this child column's child data
      auto col_child_data_size = ColumnDeviceView::extent(child) - sizeof(ColumnDeviceView);
      h_end += col_child_data_size;
      d_end += col_child_data_size;
    }
  }
  return d_children;
}

// helper function for column_device_view::create and mutable_column_device::create methods
template <typename ColumnView, typename ColumnDeviceView>
std::unique_ptr<ColumnDeviceView, std::function<void(ColumnDeviceView*)>>
create_device_view_from_view(ColumnView const& source, cudaStream_t stream)
{
  size_type num_children = source.num_children();
  // First calculate the size of memory needed to hold the
  // child columns. This is done by calling extent()
  // for each of the children.
  auto get_extent = thrust::make_transform_iterator(
    thrust::make_counting_iterator(0),
    [&source](auto i) { return ColumnDeviceView::extent(source.child(i)); });

  auto const descendant_storage_bytes =
    std::accumulate(get_extent, get_extent + num_children, std::size_t{0});

  // A buffer of CPU memory is allocated to hold the ColumnDeviceView
  // objects. Once filled, the CPU memory is copied to device memory
  // and then set into the d_children member pointer.
  std::vector<char> staging_buffer(descendant_storage_bytes);

  // Each ColumnDeviceView instance may have child objects that
  // require setting some internal device pointers before being copied
  // from CPU to device.
  rmm::device_buffer* const descendant_storage =
    new rmm::device_buffer(descendant_storage_bytes, stream);

  auto deleter = [descendant_storage](ColumnDeviceView* v) {
    v->destroy();
    delete descendant_storage;
  };

  std::unique_ptr<ColumnDeviceView, decltype(deleter)> result{
    new ColumnDeviceView(source, staging_buffer.data(), descendant_storage->data()), deleter};

  // copy the CPU memory with all the children into device memory
  CUDA_TRY(cudaMemcpyAsync(descendant_storage->data(),
                           staging_buffer.data(),
                           descendant_storage->size(),
                           cudaMemcpyDefault,
                           stream));

  CUDA_TRY(cudaStreamSynchronize(stream));

  return result;
}

}  // namespace

// Place any child objects in host memory (h_ptr) and use the device
// memory ptr (d_ptr) to set any child object pointers.
column_device_view::column_device_view(column_view source, void* h_ptr, void* d_ptr)
  : detail::column_device_view_base{source.type(),
                                    source.size(),
                                    source.head(),
                                    source.null_mask(),
                                    source.offset()},
    _num_children{source.num_children()}
{
  d_children = child_columns_to_device_array<column_view, column_device_view>(source, h_ptr, d_ptr);
}

// Construct a unique_ptr that invokes `destroy()` as it's deleter
std::unique_ptr<column_device_view, std::function<void(column_device_view*)>>
column_device_view::create(column_view source, cudaStream_t stream)
{
  size_type num_children = source.num_children();
  if (num_children == 0) {
    // Can't use make_unique since the ctor is protected
    return std::unique_ptr<column_device_view>(new column_device_view(source));
  }

  return create_device_view_from_view<column_view, column_device_view>(source, stream);
}

std::size_t column_device_view::extent(column_view const& source)
{
  auto get_extent = thrust::make_transform_iterator(
    thrust::make_counting_iterator(0), [&source](auto i) { return extent(source.child(i)); });

  return std::accumulate(
    get_extent, get_extent + source.num_children(), sizeof(column_device_view));
}

// For use with inplace-new to pre-fill memory to be copied to device
mutable_column_device_view::mutable_column_device_view(mutable_column_view source)
  : detail::column_device_view_base{source.type(),
                                    source.size(),
                                    source.head(),
                                    source.null_mask(),
                                    source.offset()},
    _num_children{source.num_children()}
{
}

mutable_column_device_view::mutable_column_device_view(mutable_column_view source,
                                                       void* h_ptr,
                                                       void* d_ptr)
  : detail::column_device_view_base{source.type(),
                                    source.size(),
                                    source.head(),
                                    source.null_mask(),
                                    source.offset()},
    _num_children{source.num_children()}
{
  d_children = child_columns_to_device_array<mutable_column_view, mutable_column_device_view>(
    source, h_ptr, d_ptr);
}

// Handle freeing children
void mutable_column_device_view::destroy() { delete this; }

// Construct a unique_ptr that invokes `destroy()` as it's deleter
std::unique_ptr<mutable_column_device_view, std::function<void(mutable_column_device_view*)>>
mutable_column_device_view::create(mutable_column_view source, cudaStream_t stream)
{
  return source.num_children() == 0
           ? std::unique_ptr<mutable_column_device_view>(new mutable_column_device_view(source))
           : create_device_view_from_view<mutable_column_view, mutable_column_device_view>(source,
                                                                                           stream);
}

std::size_t mutable_column_device_view::extent(mutable_column_view source)
{
  auto get_extent = thrust::make_transform_iterator(
    thrust::make_counting_iterator(0), [&source](auto i) { return extent(source.child(i)); });

  return std::accumulate(
    get_extent, get_extent + source.num_children(), sizeof(mutable_column_device_view));
}

}  // namespace cudf
