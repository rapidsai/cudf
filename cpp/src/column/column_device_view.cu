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
#include <utilities/error_utils.hpp>

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
  if( d_children )
    RMM_FREE(d_children,0);
  delete this;
}

// For use with inplace-new to pre-fill memory to be copied to device
column_device_view::column_device_view( column_view source, ptrdiff_t h_ptr, ptrdiff_t d_ptr )
    : detail::column_device_view_base{source.type(),       source.size(),
                                      source.head(),       source.null_mask(),
                                      source.null_count(), source.offset()},
      _num_children{source.num_children()}
{
  if( _num_children > 0 )
  {
    column_device_view* h_column = reinterpret_cast<column_device_view*>(h_ptr);
    column_device_view* d_column = reinterpret_cast<column_device_view*>(d_ptr);
    d_children = d_column;
    for( size_type idx=0; idx < _num_children; ++idx )
    { // inplace-new each child
      column_view child = source.child(idx);
      CUDF_EXPECTS( child.num_children()==0, "column grand-children not currently supported");
      new(h_column) column_device_view(child);
      h_column++;
      //d_column++;
    }
  }
}

// For use with inplace-new to pre-fill memory to be copied to device
mutable_column_device_view::mutable_column_device_view( mutable_column_view source, ptrdiff_t h_ptr, ptrdiff_t d_ptr )
    : detail::column_device_view_base{source.type(),       source.size(),
                                      source.head(),       source.null_mask(),
                                      source.null_count(), source.offset()}
{}

// Construct a unique_ptr that invokes `destroy()` as it's deleter
std::unique_ptr<column_device_view, std::function<void(column_device_view*)>> column_device_view::create(column_view source, cudaStream_t stream) {
  //size_type num_descendants{count_descendants(source)};
  //if( num_descendants > 0 )   {
  //  CUDF_FAIL("Columns with children are not currently supported.");
  // }
  auto deleter = [](column_device_view* v) { v->destroy(); };
  std::unique_ptr<column_device_view, decltype(deleter)> p{
      new column_device_view(source), deleter};
  size_type num_children = source.num_children();
  if( num_children > 0 )
  {
    // ignore grand-children right now
    RMM_ALLOC(&p->d_children, sizeof(column_device_view)*num_children, stream);
    for( size_type idx=0; idx < num_children; ++idx )
    {
      column_device_view child(source.child(idx));
      CUDF_EXPECTS( child._num_children==0, "column grand-children not currently supported");
      CUDA_TRY(cudaMemcpyAsync(p->d_children+idx, &child, sizeof(column_device_view),
                               cudaMemcpyHostToDevice, stream));
    }
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

size_type mutable_column_device_view::extent(column_view source) {
  size_type data_size = sizeof(column_device_view);
  for( size_type idx=0; idx < source.num_children(); ++idx )
    data_size += extent(source.child(idx));
  return data_size;
}


}  // namespace cudf