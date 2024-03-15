/*
 * Copyright (c) 2019-2024, NVIDIA CORPORATION.
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

#include "io/orc/aggregate_orc_metadata.hpp"
#include "io/orc/reader_impl_chunking.hpp"

#include <cudf/io/datasource.hpp>
#include <cudf/io/detail/orc.hpp>
#include <cudf/io/orc.hpp>

#include <rmm/cuda_stream_view.hpp>

#include <io/utilities/column_buffer.hpp>

#include <memory>
#include <optional>
#include <vector>

namespace cudf::io::orc::detail {

struct reader_column_meta;

/**
 * @brief Implementation for ORC reader.
 */
class reader::impl {
 public:
  /**
   * @brief Constructor from a dataset source with reader options.
   *
   * This constructor will call the other constructor with `output_size_limit` and `data_read_limit`
   * set to `0` and `output_row_granularity` set to `DEFAULT_OUTPUT_ROW_GRANULARITY`.
   *
   * @param sources Dataset sources
   * @param options Settings for controlling reading behavior
   * @param stream CUDA stream used for device memory operations and kernel launches
   * @param mr Device memory resource to use for device memory allocation
   */
  explicit impl(std::vector<std::unique_ptr<datasource>>&& sources,
                orc_reader_options const& options,
                rmm::cuda_stream_view stream,
                rmm::mr::device_memory_resource* mr);

  /**
   * @copydoc cudf::io::orc::detail::chunked_reader::chunked_reader(std::size_t, std::size_t,
   * orc_reader_options const&, rmm::cuda_stream_view, rmm::mr::device_memory_resource*)
   */
  explicit impl(std::size_t output_size_limit,
                std::size_t data_read_limit,
                std::vector<std::unique_ptr<datasource>>&& sources,
                orc_reader_options const& options,
                rmm::cuda_stream_view stream,
                rmm::mr::device_memory_resource* mr);

  /**
   * @copydoc cudf::io::orc::detail::chunked_reader::chunked_reader(std::size_t, std::size_t,
   * size_type, orc_reader_options const&, rmm::cuda_stream_view, rmm::mr::device_memory_resource*)
   */
  explicit impl(std::size_t output_size_limit,
                std::size_t data_read_limit,
                size_type output_row_granularity,
                std::vector<std::unique_ptr<datasource>>&& sources,
                orc_reader_options const& options,
                rmm::cuda_stream_view stream,
                rmm::mr::device_memory_resource* mr);

  /**
   * @copydoc cudf::io::orc::detail::reader::read
   */
  table_with_metadata read();

  /**
   * @copydoc cudf::io::chunked_orc_reader::has_next
   */
  bool has_next();

  /**
   * @copydoc cudf::io::chunked_orc_reader::read_chunk
   */
  table_with_metadata read_chunk();

 private:
  /**
   * @brief The enum indicating whether the data sources are read all at once or chunk by chunk.
   */
  enum class read_mode { READ_ALL, CHUNKED_READ };

  /**
   * @brief Perform all the necessary data preprocessing before creating an output table.
   *
   * This is the proxy to call all other data preprocessing functions, which are prerequisite
   * for generating the output.
   *
   * @param mode Value indicating if the data sources are read all at once or chunk by chunk
   */
  void prepare_data(read_mode mode);

  /**
   * @brief Perform a global preprocessing step that executes exactly once for the entire duration
   * of the reader.
   *
   * In this step, the metadata of all stripes in the data sources is parsed, and information about
   * data streams of the selected columns in all stripes are generated. If the reader has a data
   * read limit, sizes of these streams are used to split the list of all stripes into multiple
   * subsets, each of which will be read into memory in the `load_data()` step. These subsets are
   * computed such that memory usage will be kept to be around a fixed size limit.
   *
   * @param mode Value indicating if the data sources are read all at once or chunk by chunk
   */
  void global_preprocess(read_mode mode);

  /**
   * @brief Load stripes from the input data sources into memory.
   *
   * If there is a data read limit, only a subset of stripes are read at a time such that
   * their total data size does not exceed a fixed size limit. Then, the data is probed to
   * estimate its uncompressed sizes, which are in turn used to split that stripe subset into
   * smaller subsets, each of which to be decompressed and decoded in the next step
   * `decompress_and_decode()`. This is to ensure that loading data from data sources together with
   * decompression and decoding will be capped around the given data read limit.
   */
  void load_data();

  /**
   * @brief Decompress and decode stripe data in the internal buffers, and store the result into
   * an intermediate table.
   *
   * This function expects that the other preprocessing steps (`global preprocess()` and
   * `load_data()`) have already been done.
   */
  void decompress_and_decode();

  /**
   * @brief Create the output table from the intermediate table and return it along with metadata.
   *
   * @return The output table along with columns' metadata
   */
  table_with_metadata make_output_chunk();

  /**
   * @brief Create the output table metadata storing user data in source metadata.
   *
   * @return Columns' user data to output with the table read from file
   */
  table_metadata get_meta_with_user_data();

  rmm::cuda_stream_view const _stream;
  rmm::mr::device_memory_resource* const _mr;

  // Reader configs.
  struct {
    data_type timestamp_type;  // override output timestamp resolution
    bool use_index;            // enable or disable attempt to use row index for parsing
    bool use_np_dtypes;        // enable or disable the conversion to numpy-compatible dtypes
    std::vector<std::string> decimal128_columns;  // control decimals conversion

    // User specified reading rows/stripes selection.
    int64_t const skip_rows;
    std::optional<int64_t> num_read_rows;
    std::vector<std::vector<size_type>> const selected_stripes;
  } const _config;

  // Intermediate data for reading.
  std::unique_ptr<reader_column_meta> const _col_meta;  // Track of orc mapping and child details
  std::vector<std::unique_ptr<datasource>> const _sources;  // Unused but owns data for `_metadata`
  aggregate_orc_metadata _metadata;
  column_hierarchy const _selected_columns;  // Construct from `_metadata` thus declare after it
  file_intermediate_data _file_itm_data;
  chunk_read_data _chunk_read_data;

  // Intermediate data for output.
  std::unique_ptr<table_metadata> _meta_with_user_data;
  table_metadata _out_metadata;
  std::vector<std::vector<cudf::io::detail::column_buffer>> _out_buffers;

  // The default value used for subdividing the decoded table for final output.
  static inline constexpr size_type DEFAULT_OUTPUT_ROW_GRANULARITY = 10'000;
};

}  // namespace cudf::io::orc::detail
