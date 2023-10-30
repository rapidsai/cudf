/*
 * Copyright (c) 2020-2023, NVIDIA CORPORATION.
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

#pragma once

// We disable warning 611 because the `arrow::TableBatchReader` only partially
// override the `ReadNext` method of `arrow::RecordBatchReader::ReadNext`
// triggering warning 611-D from nvcc.
#ifdef __CUDACC__
#pragma nv_diag_suppress 611
#pragma nv_diag_suppress 2810
#endif
#include <arrow/api.h>
#ifdef __CUDACC__
#pragma nv_diag_default 611
#pragma nv_diag_default 2810
#endif

#include <cudf/column/column.hpp>
#include <cudf/detail/transform.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>

#include <rmm/mr/device/per_device_resource.hpp>

struct DLManagedTensor;

namespace cudf {
/**
 * @addtogroup interop_dlpack
 * @{
 * @file
 */

/**
 * @brief Convert a DLPack DLTensor into a cudf table
 *
 * The `device_type` of the DLTensor must be `kDLCPU`, `kDLCuda`, or
 * `kDLCUDAHost`, and `device_id` must match the current device. The `ndim`
 * must be set to 1 or 2. The `dtype` must have 1 lane and the bitsize must
 * match a supported `cudf::data_type`.
 *
 * @note The managed tensor is not deleted by this function.
 *
 * @throw cudf::logic_error if the any of the DLTensor fields are unsupported
 *
 * @param managed_tensor a 1D or 2D column-major (Fortran order) tensor
 * @param mr Device memory resource used to allocate the returned table's device memory
 *
 * @return Table with a copy of the tensor data
 */
std::unique_ptr<table> from_dlpack(
  DLManagedTensor const* managed_tensor,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource());

/**
 * @brief Convert a cudf table into a DLPack DLTensor
 *
 * All columns must have the same data type and this type must be numeric. The
 * columns may be nullable, but the null count must be zero. If the input table
 * is empty or has zero rows, the result will be nullptr.
 *
 * @note The `deleter` method of the returned `DLManagedTensor` must be used to
 * free the memory allocated for the tensor.
 *
 * @throw cudf::logic_error if the data types are not equal or not numeric,
 * or if any of columns have non-zero null count
 *
 * @param input Table to convert to DLPack
 * @param mr Device memory resource used to allocate the returned DLPack tensor's device memory
 *
 * @return 1D or 2D DLPack tensor with a copy of the table data, or nullptr
 */
DLManagedTensor* to_dlpack(
  table_view const& input,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource());

/** @} */  // end of group

/**
 * @addtogroup interop_arrow
 * @{
 * @file
 */

/**
 * @brief Detailed metadata information for arrow array.
 *
 * As of now this contains only name in the hierarchy of children of cudf column,
 * but in future this can be updated as per requirement.
 */
struct column_metadata {
  std::string name;                            ///< Name of the column
  std::vector<column_metadata> children_meta;  ///< Metadata of children of the column

  /**
   * @brief Construct a new column metadata object
   *
   * @param _name Name of the column
   */
  column_metadata(std::string const& _name) : name(_name) {}
  column_metadata() = default;
};

/**
 * @brief Create `arrow::Table` from cudf table `input`
 *
 * Converts the `cudf::table_view` to `arrow::Table` with the provided
 * metadata `column_names`.
 *
 * @throws cudf::logic_error if `column_names` size doesn't match with number of columns.
 *
 * @param input table_view that needs to be converted to arrow Table
 * @param metadata Contains hierarchy of names of columns and children
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param ar_mr arrow memory pool to allocate memory for arrow Table
 * @return arrow Table generated from `input`
 *
 * @note For decimals, since the precision is not stored for them in libcudf,
 * it will be converted to an Arrow decimal128 that has the widest-precision the cudf decimal type
 * supports. For example, numeric::decimal32 will be converted to Arrow decimal128 of the precision
 * 9 which is the maximum precision for 32-bit types. Similarly, numeric::decimal128 will be
 * converted to Arrow decimal128 of the precision 38.
 */
std::shared_ptr<arrow::Table> to_arrow(table_view input,
                                       std::vector<column_metadata> const& metadata = {},
                                       rmm::cuda_stream_view stream = cudf::get_default_stream(),
                                       arrow::MemoryPool* ar_mr     = arrow::default_memory_pool());

/**
 * @brief Create `arrow::Scalar` from cudf scalar `input`
 *
 * Converts the `cudf::scalar` to `arrow::Scalar`.
 *
 * @param input scalar that needs to be converted to arrow Scalar
 * @param metadata Contains hierarchy of names of columns and children
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param ar_mr arrow memory pool to allocate memory for arrow Scalar
 * @return arrow Scalar generated from `input`
 *
 * @note For decimals, since the precision is not stored for them in libcudf,
 * it will be converted to an Arrow decimal128 that has the widest-precision the cudf decimal type
 * supports. For example, numeric::decimal32 will be converted to Arrow decimal128 of the precision
 * 9 which is the maximum precision for 32-bit types. Similarly, numeric::decimal128 will be
 * converted to Arrow decimal128 of the precision 38.
 */
std::shared_ptr<arrow::Scalar> to_arrow(cudf::scalar const& input,
                                        column_metadata const& metadata = {},
                                        rmm::cuda_stream_view stream = cudf::get_default_stream(),
                                        arrow::MemoryPool* ar_mr = arrow::default_memory_pool());
/**
 * @brief Create `cudf::table` from given arrow Table input
 *
 * @param input arrow:Table that needs to be converted to `cudf::table`
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr    Device memory resource used to allocate `cudf::table`
 * @return cudf table generated from given arrow Table
 */

std::unique_ptr<table> from_arrow(
  arrow::Table const& input,
  rmm::cuda_stream_view stream        = cudf::get_default_stream(),
  rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource());

/**
 * @brief Create `cudf::scalar` from given arrow Scalar input
 *
 * @param input `arrow::Scalar` that needs to be converted to `cudf::scalar`
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr    Device memory resource used to allocate `cudf::scalar`
 * @return cudf scalar generated from given arrow Scalar
 */

std::unique_ptr<cudf::scalar> from_arrow(
  arrow::Scalar const& input,
  rmm::cuda_stream_view stream        = cudf::get_default_stream(),
  rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource());

/** @} */  // end of group
}  // namespace cudf
