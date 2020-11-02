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
#pragma once

#include <cudf/column/column_view.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>

#include <memory>

namespace cudf {
namespace detail {
/**
 * @brief Scatters the rows of the source table into a copy of the target table
 * according to a scatter map.
 *
 * Scatters values from the source table into the target table out-of-place,
 * returning a "destination table". The scatter is performed according to a
 * scatter map such that row `scatter_map[i]` of the destination table gets row
 * `i` of the source table. All other rows of the destination table equal
 * corresponding rows of the target table.
 *
 * The number of columns in source must match the number of columns in target
 * and their corresponding datatypes must be the same.
 *
 * A negative value `i` in the `scatter_map` is interpreted as `i+n`, where `n`
 * is the number of rows in the `target` table.
 *
 * If the same index appears more than once in the scatter map, the result is
 * undefined.
 *
 * @throws cudf::logic_error if `check_bounds == true` and an index exists in
 * `scatter_map` outside the range `[-n, n)`, where `n` is the number of rows in
 * the target table. If `check_bounds == false`, the behavior is undefined.
 *
 * @param source The input columns containing values to be scattered into the
 * target columns
 * @param scatter_map A non-nullable column of integral indices that maps the
 * rows in the source table to rows in the target table. The size must be equal
 * to or less than the number of elements in the source columns.
 * @param target The set of columns into which values from the source_table
 * are to be scattered
 * @param check_bounds Optionally perform bounds checking on the values of
 * `scatter_map` and throw an error if any of its values are out of bounds.
 * @param mr Device memory resource used to allocate the returned table's device memory
 * @param stream CUDA stream used for device memory operations and kernel launches.
 * @return Result of scattering values from source to target
 **/
std::unique_ptr<table> scatter(
  table_view const& source,
  column_view const& scatter_map,
  table_view const& target,
  bool check_bounds                   = false,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource(),
  cudaStream_t stream                 = 0);

/**
 * @brief Scatters a row of scalar values into a copy of the target table
 * according to a scatter map.
 *
 * Scatters values from the source row into the target table out-of-place,
 * returning a "destination table". The scatter is performed according to a
 * scatter map such that row `scatter_map[i]` of the destination table is
 * replaced by the source row. All other rows of the destination table equal
 * corresponding rows of the target table.
 *
 * The number of elements in source must match the number of columns in target
 * and their corresponding datatypes must be the same.
 *
 * If the same index appears more than once in the scatter map, the result is
 * undefined.
 *
 * @throws cudf::logic_error if `check_bounds == true` and an index exists in
 * `scatter_map` outside the range `[-n, n)`, where `n` is the number of rows in
 * the target table. If `check_bounds == false`, the behavior is undefined.
 *
 * @param source The input scalars containing values to be scattered into the
 * target columns
 * @param indices A non-nullable column of integral indices that indicate
 * the rows in the target table to be replaced by source.
 * @param target The set of columns into which values from the source_table
 * are to be scattered
 * @param check_bounds Optionally perform bounds checking on the values of
 * `scatter_map` and throw an error if any of its values are out of bounds.
 * @param mr Device memory resource used to allocate the returned table's device memory
 * @param stream CUDA stream used for device memory operations and kernel launches.
 * @return Result of scattering values from source to target
 **/
std::unique_ptr<table> scatter(
  std::vector<std::reference_wrapper<const scalar>> const& source,
  column_view const& indices,
  table_view const& target,
  bool check_bounds                   = false,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource(),
  cudaStream_t stream                 = 0);

/**
 * @copydoc cudf::boolean_mask_scatter(
                      table_view const& source, table_view const& target,
 *                    column_view const& boolean_mask,
 *                    rmm::mr::device_memory_resource *mr)
 *
 * @param stream CUDA stream used for device memory operations and kernel launches.
 */
std::unique_ptr<table> boolean_mask_scatter(table_view const& source,
                                            table_view const& target,
                                            column_view const& boolean_mask,
                                            rmm::mr::device_memory_resource* mr,
                                            cudaStream_t stream = 0);

/**
 * @copydoc cudf::boolean_mask_scatter(
 *                    std::vector<std::reference_wrapper<scalar>> const& source,
 *                    table_view const& target,
 *                    column_view const& boolean_mask,
 *                    rmm::mr::device_memory_resource *mr)
 *
 * @param stream CUDA stream used for device memory operations and kernel launches.
 */
std::unique_ptr<table> boolean_mask_scatter(
  std::vector<std::reference_wrapper<const scalar>> const& source,
  table_view const& target,
  column_view const& boolean_mask,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource(),
  cudaStream_t stream                 = 0);

}  // namespace detail
}  // namespace cudf
