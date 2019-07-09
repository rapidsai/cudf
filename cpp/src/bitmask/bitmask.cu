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
#include <cudf/bitmask/bitmask.hpp>
#include <cudf/bitmask/bitmask_device_view.hpp>
#include <cudf/bitmask/bitmask_view.hpp>
#include <cudf/types.hpp>
#include <utilities/cuda_utils.hpp>
#include <utilities/error_utils.hpp>
#include <utilities/integer_utils.hpp>

#include <rmm/mr/device_memory_resource.hpp>

namespace cudf {

namespace {

/**---------------------------------------------------------------------------*
 * @brief The size in bytes of the device memory allocation for a bitmask
 *will be rounded up to a multiple of this value.
 *---------------------------------------------------------------------------**/
constexpr std::size_t PADDING_BOUNDARY{64};

constexpr std::size_t bitmask_allocation_size(size_type number_of_bits) {
  return cudf::util::div_rounding_up_safe<size_type>(
      number_of_bits, CHAR_BIT * PADDING_BOUNDARY);
}

}  // namespace

// Allocate new device memory
bitmask::bitmask(size_type size, cudaStream_t stream,
                 rmm::mr::device_memory_resource* mr)
    : _size(size) {
  CUDF_EXPECTS(size >= 0, "Invalid size.");
  CUDF_EXPECTS(nullptr != mr, "Null memory resource.");

  _data = std::make_unique<rmm::device_buffer>(bitmask_allocation_size(_size),
                                               stream, mr);
}

// Copy from existing buffer
bitmask::bitmask(size_type size, rmm::device_buffer const& other)
    : _size{size} {
  CUDF_EXPECTS(size >= 0, "Invalid size.");
  CUDF_EXPECTS(other.size() >= bitmask_allocation_size(size),
               "Insufficiently sized buffer");
  _data = std::make_unique<rmm::device_buffer>(other);
}

// Move from existing buffer
bitmask::bitmask(size_type size, rmm::device_buffer&& other) : _size{size} {
  CUDF_EXPECTS(size >= 0, "Invalid size.");
  CUDF_EXPECTS(other.size() >= bitmask_allocation_size(size),
               "Insufficiently sized buffer");
  _data = std::make_unique<rmm::device_buffer>(other);
}

__global__ void copy_offset_bitmask(mutable_bitmask_device_view output,
                                    bitmask_device_view input, size_type offset,
                                    size_type size) {
  constexpr size_type warp_size{32};
  size_type output_index = threadIdx.x + blockIdx.x * blockDim.x;

  auto active_mask = __ballot_sync(0xFFFF'FFFF, output_index < size);

  while (output_index < size) {
    auto new_mask_element =
        __ballot_sync(active_mask, input.bit_is_set(offset + output_index));

    if (threadIdx.x % warp_size == 0) {
      output.set_element(element_index(output_index), new_mask_element);
    }

    output_index += blockDim.x * gridDim.x;
    active_mask = __ballot_sync(active_mask, output_index < size);
  }
}

// Copy from a view
bitmask::bitmask(bitmask_view source_view, cudaStream_t stream,
                 rmm::mr::device_memory_resource* mr)
    : _size{source_view.size()} {
  if (source_view.offset() == 0) {
    // If there's no offset, do a simple copy
    _data = std::make_unique<rmm::device_buffer>(
        static_cast<void const*>(source_view.data()), _size);
  } else {
    _data = std::make_unique<rmm::device_buffer>(bitmask_allocation_size(_size),
                                                 stream, mr);

    cudf::util::cuda::grid_config_1d config(_size, 256);
    copy_offset_bitmask<<<config.num_blocks, config.num_threads_per_block, 0,
                          stream>>>(this->mutable_view(), source_view,
                                    source_view.offset(), _size);

    CHECK_STREAM(stream);
  }
}

// Copy from a mutable view
bitmask::bitmask(mutable_bitmask_view source_view, cudaStream_t stream,
                 rmm::mr::device_memory_resource* mr)
    : bitmask{bitmask_view(source_view), stream, mr} {}

// Zero-copy slice
bitmask_view bitmask::slice(size_type offset, size_type slice_size) const {
  CUDF_EXPECTS(offset >= 0, "Invalid offset.");
  CUDF_EXPECTS(offset < this->size(), "Slice offset out of bounds.");
  // If the size of the slice is zero or negative, slice until the end of the
  // bitmask
  slice_size = (slice_size > 0) ? slice_size : (this->size() - offset);
  CUDF_EXPECTS((offset + slice_size) <= this->size(), "Slice out of bounds.");
  return bitmask_view{static_cast<bitmask_type const*>(_data->data()),
                      slice_size, offset};
}

// Zero-copy mutable slice
mutable_bitmask_view bitmask::mutable_slice(size_type offset,
                                            size_type slice_size) {
  CUDF_EXPECTS(offset >= 0, "Invalid offset.");
  CUDF_EXPECTS(offset < this->size(), "Slice offset out of bounds.");
  // If the size of the slice is zero or negative, slice until the end of the
  // bitmask
  slice_size = (slice_size > 0) ? slice_size : (this->size() - offset);
  CUDF_EXPECTS((offset + slice_size) <= this->size(), "Slice out of bounds.");
  return mutable_bitmask_view{static_cast<bitmask_type*>(_data->data()),
                              slice_size, offset};
}

}  // namespace cudf