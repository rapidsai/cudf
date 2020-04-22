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

#include <cudf/cudf.h>

#include <memory>
#include <string>
#include <vector>

#include <rmm/thrust_rmm_allocator.h>
#include <cudf/detail/utilities/integer_utils.hpp>
#include <cudf/legacy/table.hpp>
#include <io/utilities/datasource.hpp>
#include <io/utilities/legacy/wrapper_utils.hpp>

namespace cudf {
namespace io {
namespace orc {

// Forward internal classes
class OrcMetadata;
class OrcStreamInfo;

/**
 * @brief Implementation for ORC reader
 **/
class reader::Impl {
 public:
  /**
   * @brief Constructor from a dataset source with reader options.
   **/
  explicit Impl(std::unique_ptr<datasource> source, reader_options const &options);

  /**
   * @brief Read an entire set or a subset of data from the source and returns
   * an array of gdf_columns.
   *
   * @param[in] skip_rows Number of rows to skip from the start
   * @param[in] num_rows Number of rows to read; use `0` for all remaining data
   * @param[in] stripe Stripe index to select
   *
   * @return cudf::table Object that contains the array of gdf_columns
   **/
  table read(int skip_rows, int num_rows, int stripe);

 private:
  /**
   * @brief Helper function that populates column descriptors stream/chunk
   *
   * @param[in] chunks List of column chunk descriptors
   * @param[in] stripeinfo List of stripe metadata
   * @param[in] stripefooter List of stripe footer metadata
   * @param[in] orc2gdf Mapping of ORC columns to cuDF columns
   * @param[in] gdf2orc Mapping of cuDF columns to ORC columns
   * @param[in] types List of schema types in the dataset
   * @param[in] use_index Whether to use row index data
   * @param[out] num_dictionary_entries Number of required dictionary entries
   * @param[in,out] chunks List of column chunk descriptions
   * @param[in,out] stream_info List of column stream info
   *
   * @return size_t Size in bytes of readable stream data found
   **/
  size_t gather_stream_info(const size_t stripe_index,
                            const orc::StripeInformation *stripeinfo,
                            const orc::StripeFooter *stripefooter,
                            const std::vector<int> &orc2gdf,
                            const std::vector<int> &gdf2orc,
                            const std::vector<orc::SchemaType> types,
                            bool use_index,
                            size_t *num_dictionary_entries,
                            hostdevice_vector<orc::gpu::ColumnDesc> &chunks,
                            std::vector<OrcStreamInfo> &stream_info);

  /**
   * @brief Decompresses the stripe data, at stream granularity
   *
   * @param[in] chunks List of column chunk descriptors
   * @param[in] stripe_data List of source stripe column data
   * @param[in] decompressor Originally host decompressor
   * @param[in] stream_info List of stream to column mappings
   * @param[in] num_stripes Number of stripes making up column chunks
   * @param[in] row_groups List of row index descriptors
   * @param[in] row_index_stride Distance between each row index
   *
   * @return rmm::device_buffer Device buffer to decompressed page data
   **/
  rmm::device_buffer decompress_stripe_data(hostdevice_vector<orc::gpu::ColumnDesc> &chunks,
                                            const std::vector<rmm::device_buffer> &stripe_data,
                                            const orc::OrcDecompressor *decompressor,
                                            std::vector<OrcStreamInfo> &stream_info,
                                            size_t num_stripes,
                                            rmm::device_vector<orc::gpu::RowGroup> &row_groups,
                                            size_t row_index_stride);

  /**
   * @brief Converts the stripe column data and outputs to gdf_columns
   *
   * @param[in] chunks List of column chunk descriptors
   * @param[in] num_dicts Number of dictionary entries required
   * @param[in] skip_rows Number of rows to offset from start
   * @param[in] timezone_table Local time to UTC conversion table
   * @param[in] row_groups List of row index descriptors
   * @param[in] row_index_stride Distance between each row index
   * @param[in,out] columns List of gdf_columns
   **/
  void decode_stream_data(hostdevice_vector<orc::gpu::ColumnDesc> &chunks,
                          size_t num_dicts,
                          size_t skip_rows,
                          const std::vector<int64_t> &timezone_table,
                          rmm::device_vector<orc::gpu::RowGroup> &row_groups,
                          size_t row_index_stride,
                          const std::vector<gdf_column_wrapper> &columns);

  /**
   * @brief Align a size such that aligned 64-bit loads within a memory block
   * won't trip cuda-memcheck if the size is not a multiple of 8
   *
   * @param[in] size in bytes
   *
   * @return size_t Size aligned to the next multiple of bytes needed by orc kernels
   **/
  size_t align_size(size_t v) const { return util::round_up_safe(v, sizeof(uint64_t)); }

 private:
  std::unique_ptr<datasource> source_;
  std::unique_ptr<OrcMetadata> md_;

  std::vector<int> selected_cols_;
  bool has_timestamp_column_    = false;
  bool use_index_               = true;
  bool use_np_dtypes_           = true;
  bool decimals_as_float_       = true;
  int decimals_as_int_scale_    = -1;
  gdf_time_unit timestamp_unit_ = TIME_UNIT_NONE;
};

}  // namespace orc
}  // namespace io
}  // namespace cudf
