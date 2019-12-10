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
#include <cudf/detail/dlpack.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/utilities/error.hpp>

namespace cudf {

namespace {

data_type DLDataType_to_data_type(DLDataType type)
{
  CUDF_EXPECTS(type.lanes == 1, "Unsupported DLPack vector type");

  if (type.code == kDLInt) {
    switch (type.bits) {
      case 8:  return data_type(INT8);
      case 16: return data_type(INT16);
      case 32: return data_type(INT32);
      case 64: return data_type(INT64);
      default: CUDF_FAIL("Unsupported DLPack integer bitsize");
    }
  } else if (type.code == kDLFloat) {
    switch (type.bits) {
      case 32: return data_type(FLOAT32);
      case 64: return data_type(FLOAT64);
      default: CUDF_FAIL("Unsupported DLPack float bitsize");
    }
  } else {
    CUDF_FAIL("Unsupported DLPack unsigned type");
  }
}

}  // namespace

namespace detail {

std::vector<std::unique_ptr<column>> from_dlpack(
    DLManagedTensor const& managed_tensor,
    rmm::mr::device_memory_resource* mr,
    cudaStream_t stream)
{
  auto const& tensor = managed_tensor.dl_tensor;

  // We can copy from host or device pointers
  CUDF_EXPECTS(kDLGPU == tensor.ctx.device_type ||
               kDLCPU == tensor.ctx.device_type ||
               kDLCPUPinned == tensor.ctx.device_type, 
               "DLTensor must be GPU, CPU, or pinned type");

  // Make sure the current device ID matches the Tensor's device ID
  int device_id = 0;
  CUDA_TRY(cudaGetDevice(&device_id));
  CUDF_EXPECTS(tensor.ctx.device_id == device_id, 
               "DLTensor device ID must be current device");

  // Currently only 1D and 2D tensors are supported
  CUDF_EXPECTS(tensor.ndim > 0 && tensor.ndim <= 2,
               "DLTensor must be 1D or 2D");

  CUDF_EXPECTS(tensor.shape[0] > 0, "DLTensor first dim empty");
  CUDF_EXPECTS(tensor.shape[0] < std::numeric_limits<size_type>::max(),
               "DLTensor first dim exceeds size supported by cudf");
  if (tensor.ndim > 1) {
    CUDF_EXPECTS(tensor.shape[1] > 0, "DLTensor second dim empty");
    CUDF_EXPECTS(tensor.shape[1] < std::numeric_limits<size_type>::max(),
               "DLTensor second dim exceeds size supported by cudf");
  }

  size_t const num_columns = (tensor.ndim == 2) ?
    static_cast<size_t>(tensor.shape[1]) : 1;

  // Validate and convert data type to cudf
  data_type const dtype = DLDataType_to_data_type(tensor.dtype);

  size_t const byte_width = size_of(dtype);
  size_t const num_rows = static_cast<size_t>(tensor.shape[0]);
  size_t const bytes = num_rows * byte_width;

  // For 2D tensors, if the strides pointer is not null, then strides[1] is the
  // number of elements (not bytes) between the start of each column
  size_t const col_stride = (tensor.ndim == 2 && nullptr != tensor.strides) ?
    byte_width * tensor.strides[1] : byte_width * num_rows;

  auto tensor_data = reinterpret_cast<uintptr_t>(tensor.data) + tensor.byte_offset;

  // Allocate columns and copy data from tensor
  std::vector<std::unique_ptr<column>> output(num_columns);
  for (auto& col : output) {
    col = make_numeric_column(dtype, num_rows, UNALLOCATED, stream, mr);

    CUDA_TRY(cudaMemcpyAsync(col->mutable_view().head<void>(),
      reinterpret_cast<void*>(tensor_data), bytes, cudaMemcpyDefault, stream));

    tensor_data += col_stride;
  }

  return output;
}

}  // namespace detail

std::vector<std::unique_ptr<column>> from_dlpack(
    DLManagedTensor const& managed_tensor,
    rmm::mr::device_memory_resource* mr)
{
  return detail::from_dlpack(managed_tensor, mr);
}

}  // namespace cudf
