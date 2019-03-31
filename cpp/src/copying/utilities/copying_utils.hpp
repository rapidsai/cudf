/*
 * Copyright 2019 BlazingDB, Inc.
 *     Copyright 2019 Christian Noboa Mardini <christian@blazingdb.com>
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

#ifndef COPYING_UTILITIES_COPYING_UTILS_HPP
#define COPYING_UTILITIES_COPYING_UTILS_HPP

#include <cstdint>
#include <cuda_runtime.h>
#include "cudf/types.h"

// Forward declaration
namespace cudf {
struct column_array;
} // namespace cudf

namespace cudf {
namespace utilities {

/**
 * @brief Used to process the bitmask array. The type used to store
 * the processed bitmask into the output bitmask.
 *
 * It required to be less than the size of 'double_block_type'.
 * Used to store the processed bitmask into the output bitmask.
 */
using block_type = std::uint32_t;

/**
 * @brief Used to process the bitmask array. The type used to load
 * from the input bitmask in order to process it.
 *
 * The bitmask type that will be read from the input bitmask.
 * The read bitmask will be processed using this type.
 */
using double_block_type = std::uint64_t;

/**
 * @brief Used to create a basic type variable "gdf_size_type" in
 * the CUDA memory.
 *
 * It uses the RAII to manage the lifetime of the variable.
 * The class is not movable or copyable.
 */
class CudaVariableScope {
public:
  /**
   * Used to allocate and initialize a variable in the GPU memory.
   *
   * @param[in] stream CUDA stream used to perform several operations.
   */
  CudaVariableScope(cudaStream_t stream);

  /**
   * @brief Used to release the variable previously used in the GPU memory.
   */
  ~CudaVariableScope();

protected:
  CudaVariableScope(CudaVariableScope&&) = delete;
  CudaVariableScope(const CudaVariableScope&&) = delete;
  CudaVariableScope& operator=(CudaVariableScope&&) = delete;
  CudaVariableScope& operator=(const CudaVariableScope&) = delete;

public:
  /**
   * @brief Get the pointer of the GPU variable (already allocated in the GPU).
   */
  gdf_size_type* get_pointer();

  /**
   * @brief Used to load the variable from GPU memory and save in
   * the input parameter in CPU memory.
   *
   * @param[out] value the value of the GPU variable.
   */
  void load_value(gdf_size_type& value);

private:
  gdf_size_type* counter_; /**< Pointer to the GPU memory allocation. */
  cudaStream_t stream_;    /**< CUDA stream used to perform the operations. */
};

/**
 * @brief It contains common functionality for the different copying operations.
 */
class BaseCopying {
protected:
  /**
   * @brief It only stores the different parameters in order to be used in the
   * different methods from the base and derived class.
   *
   * @see cudf:detail::slice or @see cudf:detail::split in order to understand
   * the input parameters.
   */
  BaseCopying(gdf_column const*   input_column,
              gdf_column const*   indexes,
              cudf::column_array* output_columns,
              cudaStream_t*       streams,
              gdf_size_type       streams_size);

protected:
  /**
   * @brief Helper struct used to store the parameters for the execution of the
   * CUDA kernels.
   */
  struct KernelOccupancy {
    int grid_size{0};  /**< The number of blocks for the kernel execution. */
    int block_size{0}; /**< The number of threads per block for the kernel execution. */
  };

  /**
   * @brief Used to calculate the kernel parameters for the gdf_column operation
   * in the data kernel.
   *
   * @param[in] size The size of the data related to the operation.
   * @return KernelOccupancy Contains the calculated parameters for the execution
   * of the CUDA kernel.
   */
  KernelOccupancy calculate_kernel_data_occupancy(gdf_size_type size);

  /**
   * @brief Used to calculate the kernel parameters for the gdf_column operation
   * in the bitmask kernel.
   *
   * @param[in] size The size of the data related to the operation.
   * @return KernelOccupancy Contains the calculated parameters for the execution
   * of the CUDA kernel.
   */
  KernelOccupancy calculate_kernel_bitmask_occupancy(gdf_size_type size);

protected:
  /**
   * @brief It returns a stream according to an index value from the streams array.
   *
   * In case the streams array is empty, it returns the default stream (stream zero).
   * A fixed policy is used in order to retrieve the stream due to it uses a modulo
   * operation with the size of the streams array.
   */
  cudaStream_t get_stream(gdf_index_type index);

protected:
  /**
   * @brief Used to transform and round up the size value into the base type.
   */
  gdf_size_type round_up_size(gdf_size_type size, gdf_size_type base);

protected:
  /**
   * @brief Used to perform a common validation for the copying operation.
   */
  bool validate_inputs();

  /**
   * @brief Perform an update of the output column at the end of the operation.
   */
  void update_column(gdf_column*        output,
                     gdf_column const*  input_column,
                     CudaVariableScope& variable);

protected:
  gdf_column const*   input_column_;
  gdf_column const*   indexes_;
  cudf::column_array* output_columns_;
  cudaStream_t*       streams_;
  gdf_size_type       streams_size_;
};

}  // namespace utilities
}  // namespace cudf

#endif
