/*
 * Copyright (c) 2019, NVIDIA CORPORATION.
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

#include <rmm/rmm_api.h>
#include <rmm/thrust_rmm_allocator.h>

namespace cudf {

// Trivially copy all members but the children
column_device_view::column_device_view(column_view source)
    : detail::column_device_view_base{source.type(),       source.size(),
                                      source.head(),       source.null_mask(),
                                      source.null_count(), source.offset()},
      _num_children{source.num_children()} {}

// Free device memory allocated for children
void column_device_view::destroy() {
  delete this;
}

// Place any child objects in host memory (h_ptr) and use the device
// memory ptr (d_ptr) to set any child object pointers.
column_device_view::column_device_view( column_view source, void * h_ptr, void* d_ptr )
    : detail::column_device_view_base{source.type(),       source.size(),
                                      source.head(),       source.null_mask(),
                                      source.null_count(), source.offset()},
      _num_children{source.num_children()}
{
  size_type num_children = source.num_children();
  if( num_children > 0 )
  {
    // The beginning of the memory must be the fixed-sized column_device_view
    // struct objects in order for d_children to be used as an array.
    // Therefore, any child data is assigned past the end of this array.
    auto h_column = reinterpret_cast<column_device_view*>(h_ptr);
    auto d_column = reinterpret_cast<column_device_view*>(d_ptr);
    auto h_end = reinterpret_cast<int8_t*>(h_column + num_children);
    auto d_end = reinterpret_cast<int8_t*>(d_column + num_children);
    d_children = d_column; // set member ptr to device memory
    for( size_type idx=0; idx < _num_children; ++idx )
    { // inplace-new each child into host memory
      column_view child = source.child(idx);
      new (h_column) column_device_view(child, h_end, d_end);
      h_column++; // adv to next child
      // update the pointers for holding this child column's child data
      auto col_child_data_size = extent(child) - sizeof(column_device_view);
      h_end += col_child_data_size;
      d_end += col_child_data_size;
    }
  }
}

// Construct a unique_ptr that invokes `destroy()` as it's deleter
std::unique_ptr<column_device_view, std::function<void(column_device_view*)>>
column_device_view::create(column_view source, cudaStream_t stream) {

  size_type num_children = source.num_children();
  if (num_children == 0) {
    // Can't use make_unique since the ctor is protected
    return std::unique_ptr<column_device_view>(new column_device_view(source));
  }

  // First calculate the size of memory needed to hold the
  // child columns. This is done by calling extent()
  // for each of the children.
  auto get_extent = thrust::make_transform_iterator(
      thrust::make_counting_iterator(0),
      [&source](auto i) { return extent(source.child(i)); });

  auto const descendant_storage_bytes =
      std::accumulate(get_extent, get_extent + num_children, std::size_t{0});

  // A buffer of CPU memory is allocated to hold the column_device_view
  // objects. Once filled, the CPU memory is copied to device memory
  // and then set into the d_children member pointer.
  std::vector<char> staging_buffer(descendant_storage_bytes);

  // Each column_device_view instance may have child objects that
  // require setting some internal device pointers before being copied
  // from CPU to device.
  rmm::device_buffer* const descendant_storage =
      new rmm::device_buffer(descendant_storage_bytes, stream);

  auto deleter = [descendant_storage](column_device_view* v) {
    v->destroy();
    delete descendant_storage;
  };

  std::unique_ptr<column_device_view, decltype(deleter)> p{
      new column_device_view(source, staging_buffer.data(),
                             descendant_storage->data()),
      deleter};

  // copy the CPU memory with all the children into device memory
  CUDA_TRY(cudaMemcpyAsync(descendant_storage->data(), staging_buffer.data(),
                           descendant_storage->size(), cudaMemcpyDefault,
                           stream));

  CUDA_TRY(cudaStreamSynchronize(stream));

  return p;
}

std::size_t column_device_view::extent(column_view const& source) {
  auto get_extent = thrust::make_transform_iterator(
      thrust::make_counting_iterator(0),
      [&source](auto i) { return extent(source.child(i)); });

  return std::accumulate(get_extent, get_extent + source.num_children(),
                         sizeof(column_device_view));
}

// For use with inplace-new to pre-fill memory to be copied to device
mutable_column_device_view::mutable_column_device_view( mutable_column_view source )
    : detail::column_device_view_base{source.type(),       source.size(),
                                      source.head(),       source.null_mask(),
                                      source.null_count(), source.offset()}
{
  // TODO children may not be actually possible for mutable columns
}

mutable_column_device_view::mutable_column_device_view(
    mutable_column_view source, void* h_ptr, void* d_ptr)
    : detail::column_device_view_base{source.type(),       source.size(),
                                      source.head(),       source.null_mask(),
                                      source.null_count(), source.offset()} {
  // TODO children may not be actually possible for mutable columns
}

// Handle freeing children
void mutable_column_device_view::destroy() {
  RMM_FREE(mutable_children,0);
  delete this;
}

// Construct a unique_ptr that invokes `destroy()` as it's deleter
std::unique_ptr<mutable_column_device_view, std::function<void(mutable_column_device_view*)>>
  mutable_column_device_view::create(mutable_column_view source, cudaStream_t stream) {
  // TODO children may not be actually possible for mutable columns
  auto deleter = [](mutable_column_device_view* v) { v->destroy(); };
  std::unique_ptr<mutable_column_device_view, decltype(deleter)> p{
      new mutable_column_device_view(source), deleter};
  return p;
}

std::size_t mutable_column_device_view::extent(mutable_column_view source) {
  auto get_extent = thrust::make_transform_iterator(
      thrust::make_counting_iterator(0),
      [&source](auto i) { return extent(source.child(i)); });

  return std::accumulate(get_extent, get_extent + source.num_children(),
                         sizeof(mutable_column_device_view));
}


}  // namespace cudf
