/*
 * Copyright (c) 2025, NVIDIA CORPORATION.
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

#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/detail/reshape.hpp>
#include <cudf/reshape.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_uvector.hpp>

#include <cub/device/device_memcpy.cuh>
#include <cuda/functional>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>

namespace cudf {
namespace detail {
namespace {

template <typename T>
void _table_to_array(table_view const& input, void* output, rmm::cuda_stream_view stream)
{
  auto const num_columns = input.num_columns();
  auto const num_rows    = input.num_rows();
  auto const item_size   = sizeof(T);
  auto* base_ptr         = static_cast<cuda::std::byte*>(output);

  CUDF_EXPECTS(num_columns > 0, "Must have at least one column.");
  CUDF_EXPECTS(output != nullptr, "Output pointer cannot be null.");

  rmm::device_uvector<void*> d_srcs(num_columns, stream);
  rmm::device_uvector<void*> d_dsts(num_columns, stream);

  std::vector<void const*> h_srcs(num_columns);
  std::vector<void*> h_dsts(num_columns);

  for (int i = 0; i < num_columns; ++i) {
    auto const& col = input.column(i);
    CUDF_EXPECTS(col.type() == input.column(0).type(), "All columns must have the same dtype");
    CUDF_EXPECTS(col.null_count() == 0, "All columns must be non-nullable or contain no nulls");

    h_srcs[i] = static_cast<void const*>(col.data<T>());
    h_dsts[i] = static_cast<void*>(base_ptr + i * item_size * num_rows);
  }

  CUDF_CUDA_TRY(cudaMemcpyAsync(d_srcs.data(),
                                h_srcs.data(),
                                sizeof(void*) * num_columns,
                                cudaMemcpyHostToDevice,
                                stream.value()));
  CUDF_CUDA_TRY(cudaMemcpyAsync(d_dsts.data(),
                                h_dsts.data(),
                                sizeof(void*) * num_columns,
                                cudaMemcpyHostToDevice,
                                stream.value()));

  thrust::constant_iterator<size_t> sizes(static_cast<size_t>(item_size * num_rows));

  void* d_temp_storage      = nullptr;
  size_t temp_storage_bytes = 0;
  cub::DeviceMemcpy::Batched(d_temp_storage,
                             temp_storage_bytes,
                             d_srcs.begin(),
                             d_dsts.begin(),
                             sizes,
                             num_columns,
                             stream.value());

  rmm::device_buffer temp_storage(temp_storage_bytes, stream);

  cub::DeviceMemcpy::Batched(temp_storage.data(),
                             temp_storage_bytes,
                             d_srcs.begin(),
                             d_dsts.begin(),
                             sizes,
                             num_columns,
                             stream.value());
}

struct TableToArrayDispatcher {
  table_view const& input;
  void* output;
  rmm::cuda_stream_view stream;

  template <typename T, CUDF_ENABLE_IF(is_fixed_width<T>() || is_fixed_point<T>())>
  void operator()() const
  {
    if constexpr (is_fixed_point<T>()) {
      using StorageType = cudf::device_storage_type_t<T>;
      _table_to_array<StorageType>(input, output, stream);
    } else {
      _table_to_array<T>(input, output, stream);
    }
  }

  template <typename T, CUDF_ENABLE_IF(!is_fixed_width<T>() && !is_fixed_point<T>())>
  void operator()() const
  {
    CUDF_FAIL("Unsupported dtype");
  }
};

}  // namespace

void table_to_array(table_view const& input,
                    void* output,
                    data_type output_dtype,
                    rmm::cuda_stream_view stream)
{
  CUDF_EXPECTS(output != nullptr, "Output pointer cannot be null.");
  CUDF_EXPECTS(input.num_columns() > 0, "Input must have at least one column.");

  cudf::type_dispatcher(output_dtype, TableToArrayDispatcher{input, output, stream});
}

}  // namespace detail

void table_to_array(table_view const& input,
                    void* output,
                    data_type output_dtype,
                    rmm::cuda_stream_view stream)
{
  CUDF_FUNC_RANGE();
  cudf::detail::table_to_array(input, output, output_dtype, stream);
}

}  // namespace cudf
