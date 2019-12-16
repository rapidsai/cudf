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
  // TODO Needs to handle grand-children
  RMM_FREE(d_children,0);
  delete this;
}

// Place any child objects in host memory (h_ptr) and use the device
// memory ptr (d_ptr) to set any child object pointers.
column_device_view::column_device_view( column_view source, ptrdiff_t h_ptr, ptrdiff_t d_ptr )
    : detail::column_device_view_base{source.type(),       source.size(),
                                      source.head(),       source.null_mask(),
                                      source.null_count(), source.offset()},
      _num_children{source.num_children()}
{
  if( count_descendants(source) > _num_children ) {
    CUDF_FAIL("Columns with grand-children are not currently supported.");
  }
  if( _num_children > 0 )
  {
    column_device_view* h_column = reinterpret_cast<column_device_view*>(h_ptr);
    column_device_view* d_column = reinterpret_cast<column_device_view*>(d_ptr);
    d_children = d_column;
    for( size_type idx=0; idx < _num_children; ++idx )
    { // inplace-new each child
      column_view child = source.child(idx);
      new(h_column) column_device_view(child);
      h_column++;
    }
  }
}

// Construct a unique_ptr that invokes `destroy()` as it's deleter
std::unique_ptr<column_device_view, std::function<void(column_device_view*)>> column_device_view::create(column_view source, cudaStream_t stream) {
  size_type num_children = source.num_children();
  if( count_descendants(source) > num_children ) {
    CUDF_FAIL("Columns with grand-children are not currently supported.");
  }
  auto deleter = [](column_device_view* v) { v->destroy(); };
  std::unique_ptr<column_device_view, decltype(deleter)> p{
      new column_device_view(source), deleter};

  if( num_children > 0 )
  {
    // create device memory for the children
    RMM_TRY(RMM_ALLOC(&p->d_children, sizeof(column_device_view)*num_children, stream));
    // build the children into CPU memory first
    std::vector<uint8_t> buffer(sizeof(column_device_view)*num_children);
    auto h_ptr = buffer.data();
    for( size_type idx=0; idx < num_children; ++idx )
    {
      // create device-view from view
      column_device_view child(source.child(idx));
      // copy child into buffer
      memcpy(h_ptr, &child, sizeof(column_device_view));
      // point to the next array slot
      h_ptr += sizeof(column_device_view);
    }
    // copy the CPU memory with the children into device memory
    CUDA_TRY(cudaMemcpyAsync(p->d_children, buffer.data(), num_children*sizeof(column_device_view),
                              cudaMemcpyHostToDevice, stream));
    p->_num_children = num_children;
    cudaStreamSynchronize(stream);
  }
  return p;
}

size_type column_device_view::extent(column_view source) {
  size_type data_size = sizeof(column_device_view);
  for( size_type idx=0; idx < source.num_children(); ++idx )
    data_size += extent(source.child(idx));
  return data_size;
}

// For use with inplace-new to pre-fill memory to be copied to device
mutable_column_device_view::mutable_column_device_view( mutable_column_view source )
    : detail::column_device_view_base{source.type(),       source.size(),
                                      source.head(),       source.null_mask(),
                                      source.null_count(), source.offset()}
{
  // TODO children may not be actually possible for mutable columns
}

mutable_column_device_view::mutable_column_device_view( mutable_column_view source, ptrdiff_t h_ptr, ptrdiff_t d_ptr )
    : detail::column_device_view_base{source.type(),       source.size(),
                                      source.head(),       source.null_mask(),
                                      source.null_count(), source.offset()}
{
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

size_type mutable_column_device_view::extent(mutable_column_view source) {
  size_type data_size = sizeof(mutable_column_device_view);
  for( size_type idx=0; idx < source.num_children(); ++idx )
    data_size += extent(source.child(idx));
  return data_size;
}


}  // namespace cudf
