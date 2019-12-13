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
#include <cudf/utilities/type_dispatcher.hpp>
#include <cudf/utilities/error.hpp>

#include <dlpack/dlpack.h>

#include <algorithm>

namespace cudf {

namespace {

struct get_column_data_impl {
  template <typename T>
  void const* operator()(column_view const& col) { return col.data<T>(); }
};

void const* get_column_data(column_view const& col)
{
  return experimental::type_dispatcher(col.type(), get_column_data_impl{}, col);
}

data_type DLDataType_to_data_type(DLDataType type)
{
  CUDF_EXPECTS(type.lanes == 1, "Unsupported DLPack vector type");

  if (type.code == kDLInt) {
    switch (type.bits) {
      case 8:  return data_type(INT8);
      case 16: return data_type(INT16);
      case 32: return data_type(INT32);
      case 64: return data_type(INT64);
      default: CUDF_FAIL("Unsupported bitsize for kDLInt");
    }
  } else if (type.code == kDLUInt) {
    switch (type.bits)
    {
    case 8:  return data_type(BOOL8);
    default: CUDF_FAIL("Unsupported bitsize for kDLUInt");
    }
  } else if (type.code == kDLFloat) {
    switch (type.bits) {
      case 32: return data_type(FLOAT32);
      case 64: return data_type(FLOAT64);
      default: CUDF_FAIL("Unsupported bitsize for kDLFloat");
    }
  } else {
    CUDF_FAIL("Invalid DLPack type code");
  }
}

struct data_type_to_DLDataType_impl {
  template <typename T, std::enable_if_t<is_numeric<T>()>* = nullptr>
  DLDataType operator()() {
    uint8_t const bits{sizeof(T) * 8};
    uint16_t const lanes{1};
    if (std::is_floating_point<T>::value) {
      return DLDataType{kDLFloat, bits, lanes};
    } else if (std::is_signed<T>::value) {
      return DLDataType{kDLInt, bits, lanes};
    } else {
      return DLDataType{kDLUInt, bits, lanes};
    }
  }

  template <typename T, std::enable_if_t<not is_numeric<T>()>* = nullptr>
  DLDataType operator()() {
    CUDF_FAIL("Conversion of non-numeric types to DLPack is unsupported");
  }
};

DLDataType data_type_to_DLDataType(data_type type)
{
  return experimental::type_dispatcher(type, data_type_to_DLDataType_impl{});
}

// Context object to own memory allocated for DLManagedTensor
struct dltensor_context {
  int64_t shape[2];
  int64_t strides[2];
  rmm::device_buffer buffer;

  static void deleter(DLManagedTensor* arg)
  {
    auto context = static_cast<dltensor_context*>(arg->manager_ctx);
    delete context;
    delete arg;
  }
};

}  // namespace

namespace detail {

std::unique_ptr<experimental::table> from_dlpack(
    DLManagedTensor const* managed_tensor,
    rmm::mr::device_memory_resource* mr,
    cudaStream_t stream)
{
  CUDF_EXPECTS(nullptr != managed_tensor, "managed_tensor is null");
  auto const& tensor = managed_tensor->dl_tensor;

  // We can copy from host or device pointers
  CUDF_EXPECTS(kDLGPU == tensor.ctx.device_type ||
               kDLCPU == tensor.ctx.device_type ||
               kDLCPUPinned == tensor.ctx.device_type, 
               "DLTensor must be GPU, CPU, or pinned type");

  // Make sure the current device ID matches the Tensor's device ID
  if (tensor.ctx.device_type != kDLCPU) {
    int device_id = 0;
    CUDA_TRY(cudaGetDevice(&device_id));
    CUDF_EXPECTS(tensor.ctx.device_id == device_id, 
                 "DLTensor device ID must be current device");
  }

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
  std::vector<std::unique_ptr<column>> columns(num_columns);
  for (auto& col : columns) {
    col = make_numeric_column(dtype, num_rows, UNALLOCATED, stream, mr);

    CUDA_TRY(cudaMemcpyAsync(col->mutable_view().head<void>(),
      reinterpret_cast<void*>(tensor_data), bytes, cudaMemcpyDefault, stream));

    tensor_data += col_stride;
  }

  return std::make_unique<experimental::table>(std::move(columns));
}

DLManagedTensor* to_dlpack(table_view const& input,
    rmm::mr::device_memory_resource* mr, cudaStream_t stream)
{
  auto const num_rows = input.num_rows();
  auto const num_cols = input.num_columns();
  if (num_rows == 0) {
    return nullptr;
  }

  // Ensure that type is convertible to DLDataType
  data_type const type = input.column(0).type();
  DLDataType const dltype = data_type_to_DLDataType(type);

  // Ensure all columns are the same type
  CUDF_EXPECTS(std::all_of(input.begin(), input.end(),
    [type](auto const& col) { return col.type() == type; }),
    "All columns required to have same data type");

  // Ensure none of the columns have nulls
  CUDF_EXPECTS(std::none_of(input.begin(), input.end(),
    [](auto const& col) { return col.has_nulls(); }),
    "Input required to have null count zero");

  auto managed_tensor = std::make_unique<DLManagedTensor>();
  auto context = std::make_unique<dltensor_context>();

  DLTensor& tensor = managed_tensor->dl_tensor;
  tensor.dtype = dltype;

  tensor.ndim = (num_cols > 1) ? 2 : 1;
  tensor.shape = context->shape;
  tensor.shape[0] = num_rows;
  if (tensor.ndim > 1) {
    tensor.shape[1] = num_cols;
    tensor.strides = context->strides;
    tensor.strides[0] = 1;
    tensor.strides[1] = num_rows;
  }

  CUDA_TRY(cudaGetDevice(&tensor.ctx.device_id));
  tensor.ctx.device_type = kDLGPU;

  // If there is only one column, then a 1D tensor can just copy the pointer
  // to the data in the column, and the deleter should not delete the original
  // data. However, this is inconsistent with the 2D cases where we must do a
  // copy of each column's data into the dense tensor array. Also, if we don't
  // copy, then the original column data could be changed, which would change
  // the contents of the tensor, which might be surprising or cause issues.
  // Therefore, for now we ALWAYS do a copy of the data. If this becomes
  // a performance issue we can reevaluate in the future.

  size_t const stride_bytes = num_rows * size_of(type);
  size_t const total_bytes = stride_bytes * num_cols;

  context->buffer = rmm::device_buffer(total_bytes, stream, mr);
  tensor.data = context->buffer.data();

  auto tensor_data = reinterpret_cast<uintptr_t>(tensor.data);
  for (auto const& col : input) {
    CUDA_TRY(cudaMemcpyAsync(reinterpret_cast<void*>(tensor_data),
      get_column_data(col), stride_bytes, cudaMemcpyDefault, stream));
    tensor_data += stride_bytes;
  }

  // Defer ownership of managed tensor to caller
  managed_tensor->deleter = dltensor_context::deleter;
  managed_tensor->manager_ctx = context.release();
  return managed_tensor.release();
}

}  // namespace detail

std::unique_ptr<experimental::table> from_dlpack(
    DLManagedTensor const* managed_tensor,
    rmm::mr::device_memory_resource* mr)
{
  return detail::from_dlpack(managed_tensor, mr);
}

DLManagedTensor* to_dlpack(table_view const& input,
    rmm::mr::device_memory_resource* mr)
{
  return detail::to_dlpack(input, mr);
}

}  // namespace cudf
