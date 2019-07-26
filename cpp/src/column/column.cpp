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

#include <cudf/column/column.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/utils/type_dispatcher.hpp>

#include <rmm/device_buffer.hpp>
#include <rmm/mr/device_memory_resource.hpp>

#include <algorithm>
#include <vector>

namespace cudf {

// Move constructor
column::column(column &&other)
    : _type{other._type},
      _size{other._size},
      _data{std::move(other._data)},
      _null_mask{std::move(other._null_mask)},
      _null_count{other._null_count},
      _children{std::move(other._children)} {
  other._size = 0;
  other._null_count = 0;
  other._type = data_type{EMPTY};
}

// Create immutable view
column_view column::view() const {
  // Create views of children
  std::vector<column_view> child_views(_children.size());
  std::copy(begin(_children), end(_children), begin(child_views));

  return column_view{_type,
                     _size,
                     _data.data(),
                     static_cast<bitmask_type const *>(_null_mask.data()),
                     _null_count,
                     0,
                     std::move(child_views)};
}

// Create mutable view
mutable_column_view column::mutable_view() {
  // create views of children
  std::vector<mutable_column_view> child_views(_children.size());
  std::copy(begin(_children), end(_children), begin(child_views));

  // Store the old null count
  auto current_null_count = _null_count;

  // The elements of a column could be changed through a `mutable_column_view`,
  // therefore the existing `null_count` is no longer valid. Reset it to
  // `UNKONWN_NULL_COUNT` forcing it to be recomputed on the next invocation of
  // `null_count()`.
  _null_count = UNKNOWN_NULL_COUNT;

  return mutable_column_view{_type,
                             _size,
                             _data.data(),
                             static_cast<bitmask_type *>(_null_mask.data()),
                             current_null_count,
                             0,
                             std::move(child_views)};
}

// If the null count is known, return it. Else, compute and return it
size_type column::null_count() const {
  if (_null_count > UNKNOWN_NULL_COUNT) {
    return _null_count;
  } else {
    // If the null mask isn't allocated, then we can return 0
    if (0 == _null_mask.size()) {
      return 0;
    }
    CUDF_FAIL(
        "On-demand computation of null count is not currently supported.");
  }
}

void column::set_null_count(size_type new_null_count) {
  if (new_null_count > 0) {
    CUDF_EXPECTS(_null_mask.size() > 0, "Invalid null count.");
  }
  _null_count = new_null_count;
}

/**---------------------------------------------------------------------------*
 * @brief Copies the bits starting at the specified offset from a source
 * bitmask into the destination bitmask.
 *
 * Bit `i` in `destination` will be equal to bit `i + offset` from `source`.
 *
 * @param destination The mask to copy into
 * @param source The mask to copy from
 * @param bit_offset The offset into `source` from which to begin the copy
 * @param number_of_bits The number of bits to copy
 *---------------------------------------------------------------------------**/
//__global__ void copy_offset_bitmask(mutable_bitmask_device_view destination,
//                                    bitmask_device_view source,
//                                    size_type bit_offset,
//                                    size_type number_of_bits) {
//  constexpr size_type warp_size{32};
//  size_type destination_index = threadIdx.x + blockIdx.x * blockDim.x;
//
//  auto active_mask =
//      __ballot_sync(0xFFFF'FFFF, destination_index < number_of_bits);
//
//  while (destination_index < number_of_bits) {
//    bitmask_type const new_mask_element = __ballot_sync(
//        active_mask, source.bit_is_set(bit_offset + destination_index));
//
//    if (threadIdx.x % warp_size == 0) {
//      destination.set_element(element_index(destination_index),
//                              new_mask_element);
//    }
//
//    destination_index += blockDim.x * gridDim.x;
//    active_mask =
//        __ballot_sync(active_mask, destination_index < number_of_bits);
//  }
//}

// Copy from a view
column::column(column_view view, cudaStream_t stream,
               rmm::mr::device_memory_resource *mr) {
  // if (source_view.offset() == 0) {
  //  // If there's no offset, do a simple copy
  //  _data = std::make_unique<rmm::device_buffer>(
  //      static_cast<void const *>(source_view.data()), _size);
  //} else {
  //  // If there's a non-zero offset, need to handle offset bitmask elements
  //  _data = std::make_unique<rmm::device_buffer>(
  //      bitmask_allocation_size_bytes(_size), stream, mr);

  //  cudf::util::cuda::grid_config_1d config(_size, 256);
  //  copy_offset_bitmask<<<config.num_blocks, config.num_threads_per_block, 0,
  //                        stream>>>(this->mutable_view(), source_view,
  //                                  source_view.offset(), _size);

  //  CHECK_STREAM(stream);
}

}  // namespace cudf