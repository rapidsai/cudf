/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2024, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cudf/contiguous_split.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <rmm/cuda_stream_view.hpp>

namespace CUDF_EXPORT cudf {
namespace detail {

/**
 * @copydoc cudf::contiguous_split
 *
 * @param stream CUDA stream used for device memory operations and kernel launches.
 **/
std::vector<packed_table> contiguous_split(cudf::table_view const& input,
                                           std::vector<size_type> const& splits,
                                           rmm::cuda_stream_view stream,
                                           rmm::device_async_resource_ref mr);

/**
 * @copydoc cudf::pack
 *
 * @param stream Optional CUDA stream on which to execute kernels
 **/
packed_columns pack(cudf::table_view const& input,
                    rmm::cuda_stream_view stream,
                    rmm::device_async_resource_ref mr);

// opaque implementation of `metadata_builder` since it needs to use
// `serialized_column`, which is only defined in pack.cpp
class metadata_builder_impl;

/**
 * @brief Helper class that creates packed column metadata.
 *
 * This class is an interface to the opaque metadata that is used to
 * describe `contiguous_split` and `pack` results.
 */
class metadata_builder {
 public:
  /**
   * @brief Construct a new metadata_builder.
   *
   * @param num_root_columns is the number of top-level columns
   */
  explicit metadata_builder(size_type const num_root_columns);

  /**
   * @brief Destructor that will be implemented as default, required because metadata_builder_impl
   * is incomplete at this stage.
   */
  ~metadata_builder();

  /**
   * @brief Add a column to this metadata builder.
   *
   * Callers must call this function for the parent column and followed by any children,
   * in the order maintained in the column/column_view.
   *
   * Example: given a table with a nested column "a" with 2 children, and a non-nested column "b":
   *
   *   1) add_column_info_to_meta(col_a)
   *   2) add_column_info_to_meta(col_a_child_1)
   *   3) add_column_info_to_meta(col_a_child_2)
   *   4) add_column_info_to_meta(col_b)
   *
   * @param col_type column data type
   * @param col_size column row count
   * @param col_null_count column null count
   * @param data_offset data offset from the column's base ptr,
   *                    or -1 for an empty column
   * @param null_mask_offset null mask offset from the column's base ptr,
   *                    or -1 for a column that isn't nullable
   * @param num_children number of children columns
   */
  void add_column_info_to_meta(data_type const col_type,
                               size_type const col_size,
                               size_type const col_null_count,
                               int64_t const data_offset,
                               int64_t const null_mask_offset,
                               size_type const num_children);

  /**
   * @brief Builds the opaque metadata for all added columns.
   *
   * @returns A vector containing the serialized column metadata
   */
  [[nodiscard]] std::vector<uint8_t> build() const;

  /**
   * @brief Clear the internal buffer containing all added metadata.
   */
  void clear();

 private:
  std::unique_ptr<metadata_builder_impl> impl;
};

/**
 * @copydoc pack_metadata
 * @param builder The reusable builder object to create packed column metadata.
 */
std::vector<uint8_t> pack_metadata(table_view const& table,
                                   uint8_t const* contiguous_buffer,
                                   size_t buffer_size,
                                   metadata_builder& builder);

}  // namespace detail
}  // namespace CUDF_EXPORT cudf
