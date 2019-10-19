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
#include "transpose.hpp"
#include <cudf/table/table_device_view.cuh>
#include <utilities/nvtx/nvtx_utils.hpp>
#include <cudf/utilities/type_dispatcher.hpp>
#include <cudf/utilities/traits.hpp>
#include <cudf/copying.hpp> // TODO use internal <copying/copy.hpp>

//#include <rmm/thrust_rmm_allocator.h>
//#include <cudf/cudf.h>
//#include <cub/cub.cuh>
//#include <memory>
//#include <algorithm>

namespace cudf {

namespace {

constexpr int WARP_SIZE = 32;
constexpr int MAX_GRID_SIZE = (1<<16)-1;

/**
 * @brief Transposes the values from ncols x nrows input columns to
 *  nrows x ncols output columns
 * 
 * @tparam T  Datatype of values pointed to by the views
 * @param input[in]  Device view of input columns' data
 * @param output[out]  Mutable device view of pre-allocated output columns' data
 */
template <typename T>
__global__
void gpu_transpose(table_device_view const input, mutable_table_device_view output)
{
  size_type x = blockIdx.x * blockDim.x + threadIdx.x;
  size_type y = blockIdx.y * blockDim.y + threadIdx.y;

  size_type stride_x = blockDim.x * gridDim.x;
  size_type stride_y = blockDim.y * gridDim.y;

  for(size_type i = x; i < input.num_columns(); i += stride_x)
  {
    for(size_type j = y; j < input.num_rows(); j += stride_y)
    {
      output.column(j).element<T>(i) = input.column(i).element<T>(j);
    }
  }
}

/**
 * @brief Transposes the validity mask
 * 
 * @param[in] in_cols_valid  pointers to the validity mask of the input columns
 * @param[out] out_cols_valid  pointers to the pre-allocated validity mask of
 *  the output columns
 * @param[out] out_cols_null_count  array of per output-row null counts
 * @param[in] ncols  number of columns in input table
 * @param[in] nrows  number of rows in input table
 */
__global__
void gpu_transpose_valids(table_device_view const input, mutable_table_device_view output)
// TODO profile these two implementations
#if 1
{
  size_type x = blockIdx.x * blockDim.x + threadIdx.x;
  size_type y = blockIdx.y * blockDim.y + threadIdx.y;

  size_type stride_x = blockDim.x * gridDim.x;
  size_type stride_y = blockDim.y * gridDim.y;

  for(size_type i = x; i < input.num_columns(); i += stride_x)
  {
    for(size_type j = y; j < input.num_rows(); j += stride_y)
    {
      if (input.column(i).is_valid(j)) {
        output.column(j).set_valid(i);
      } else {
        output.column(j).set_null(i);
      }
    }
  }
}
#else
{
  constexpr uint32_t BITS_PER_MASK{sizeof(bitmask_type) * 8};

  size_type x = blockIdx.x * blockDim.x + threadIdx.x;
  size_type y = blockIdx.y * blockDim.y + threadIdx.y;

  size_type stride_x = blockDim.x * gridDim.x;
  size_type stride_y = blockDim.y * gridDim.y;

  size_type i = x;
  size_type j = y;
  auto active_threads = __ballot_sync(0xffffffff, i < input.num_columns());
  while(i < input.num_columns())
  {
    j = y;
    while(j < input.num_rows())
    {
      bool const input_is_valid{input.column(i).is_valid(j)};
      bitmask_type const result_mask{__ballot_sync(active_threads, input_is_valid)};

      bitmask_type* const __restrict__ out_mask32 = output.column(j).null_mask();

      cudf::size_type const out_location = i / BITS_PER_MASK;

      // Only one thread writes output
      if (0 == threadIdx.x % warpSize) {
        out_mask32[out_location] = result_mask;
        // TODO we can write the null count to an additional device buffer
        //int num_nulls = __popc(active_threads) - __popc(result_mask);
        //atomicAdd(out_cols_null_count + j, num_nulls);
      }
      
      j += stride_y;
    }
    i += stride_x;
    active_threads = __ballot_sync(active_threads, i < input.num_columns());
  }
}
#endif

// TODO: refactor and separate `valids` kernel launch into another function.
// Should not need to pass `has_null`
struct launch_kernel{
  template <typename T>
  void operator()(
    table_view const& input,
    mutable_table_view& output,
    bool has_null,
    cudaStream_t stream)
  {
    // Copy input columns `data` and `valid` pointers to device
    auto device_input = table_device_view::create(input, stream);
    auto device_output = mutable_table_device_view::create(output, stream);

    dim3 dimBlock(WARP_SIZE, WARP_SIZE);
    dim3 dimGrid(std::min((input.num_columns() + WARP_SIZE - 1) / WARP_SIZE, MAX_GRID_SIZE),
                 std::min((input.num_rows() + WARP_SIZE - 1) / WARP_SIZE, MAX_GRID_SIZE));

    gpu_transpose<T><<<dimGrid, dimBlock, 0, stream>>>(*device_input, *device_output);

    if (has_null){
      gpu_transpose_valids<<<dimGrid, dimBlock, 0, stream>>>(*device_input, *device_output);

      // Force null counts to be recomputed next time they are queried
      for (auto& column : output) {
        column.set_null_count(UNKNOWN_NULL_COUNT);
      }
    }

    // Synchronize before return so we don't cut short the lifetime of our device_views
    CUDA_TRY(cudaStreamSynchronize(stream));
  }
};

}  // namespace

namespace detail {

std::unique_ptr<experimental::table> transpose(table_view const& input,
  rmm::mr::device_memory_resource* mr, cudaStream_t stream)
{
  auto const input_ncols = input.num_columns();
  auto const input_nrows = input.num_rows();

  // If there are no rows in the input, return successfully
  if (input_ncols == 0 || input_nrows == 0) {
    // NOTE this returns a table with the same shape as the input, not transposed
    // TODO this could take stream as a parameter
    return experimental::empty_like(input/*, stream*/);
  }

  // Check datatype homogeneity
  auto const dtype = input.column(0).type();
  for (auto const& col : input) {
    CUDF_EXPECTS(dtype == col.type(), "Column type mismatch");
  }

  // TODO does this need to support non-fixed-width tables?
  CUDF_EXPECTS(is_fixed_width(dtype), "Invalid, non-fixed-width type.");

  nvtx::range_push("CUDF_TRANSPOSE", nvtx::color::GREEN);

  // Check if there are nulls to be processed
  bool const has_null = has_nulls(input);
  auto const allocation_policy = has_null ? experimental::mask_allocation_policy::ALWAYS
    : experimental::mask_allocation_policy::NEVER;

  auto const& output_ncols = input_nrows;
  auto const& output_nrows = input_ncols;

  // Allocate output table with transposed shape
  std::vector<std::unique_ptr<column>> out_columns;
  out_columns.reserve(output_ncols);
  for (size_type i = 0; i < output_ncols; ++i) {
    // TODO this could take stream as a parameter
    out_columns.push_back(experimental::allocate_like(input.column(0), output_nrows,
      allocation_policy, mr/*, stream*/));
  }
  auto output = std::make_unique<experimental::table>(std::move(out_columns));
  auto output_view = output->mutable_view();

  experimental::type_dispatcher(dtype, launch_kernel{}, input, output_view, has_null, stream);

  nvtx::range_pop();
  return output;
}

}  // namespace detail

std::unique_ptr<experimental::table> transpose(table_view const& input,
                                 rmm::mr::device_memory_resource* mr)
{
  return detail::transpose(input, mr);
}

}  // namespace cudf
