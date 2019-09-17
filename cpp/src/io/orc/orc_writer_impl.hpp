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

#include <fstream>
#include <memory>
#include <string>
#include <vector>

#include "orc.h"
#include "orc_gpu.h"

#include <io/utilities/wrapper_utils.hpp>

#include <cudf/cudf.h>
#include <cudf/legacy/table.hpp>
#include <utilities/error_utils.hpp>

namespace cudf {
namespace io {
namespace orc {

// Forward internal classes
class ProtobufWriter;

/**
 * @brief Implementation for ORC writer
 **/
class writer::Impl {
  static constexpr const char* MAGIC = "ORC";

  // ORC datasets are divided into fixed size, independent stripes
  // TODO: Currently fixed to limit of 64 MB to match Apache default
  const uint32_t MAX_STRIPE_SIZE = 64 * 1024 * 64;

  // ORC rows are divided into groups and assigned indexes for faster seeking
  // TODO: Currently fixed to limit of 10000 rows to match Apache default
  const uint32_t ROW_INDEX_STRIDE = 10000;

 public:
  /**
   * @brief Constructor with writer options.
   **/
  explicit Impl(std::string filepath, writer_options const& options);

  /**
   * @brief Write an entire dataset to ORC format
   *
   * @param[in] table Table of columns
   **/
  void write(const cudf::table& table);

 private:
  /**
   * @brief Write an entire dataset to ORC format
   *
   * @param[in] table Table of columns
   **/
  void build_dictionary_chunks(
      const cudf::table& table, size_t num_rows,
      const std::vector<int>& str_col_ids,
      const device_buffer<uint32_t>& dict_data,
      const device_buffer<uint32_t>& dict_index,
      std::vector<device_buffer<std::pair<const char*, size_t>>>& indices,
      hostdevice_vector<gpu::DictionaryChunk>& dict);

  /**
   * @brief Write an entire dataset to ORC format
   *
   * @param[in] table Table of columns
   **/
  void build_stripe_dictionaries(
      size_t num_rows, const std::vector<int>& str_col_ids,
      const std::vector<uint32_t>& stripe_list,
      const hostdevice_vector<gpu::DictionaryChunk>& dict,
      const device_buffer<uint32_t>& dict_index,
      hostdevice_vector<gpu::StripeDictionary>& stripe_dict);

  /**
   * @brief Write an entire dataset to ORC format
   *
   * @param[in] table Table of columns
   **/
  StripeFooter init_stripe_footer(
      const cudf::table& table, std::vector<Stream>& streams,
      const std::vector<uint32_t>& stripe_list,
      const hostdevice_vector<gpu::DictionaryChunk>& dict,
      const hostdevice_vector<gpu::StripeDictionary>& stripe_dict,
      const std::vector<int>& str_col_map, std::vector<int32_t>& str_ids,
      std::vector<SchemaType>& types);

  /**
   * @brief Write an entire dataset to ORC format
   *
   * @param[in] table Table of columns
   **/
  std::vector<StripeInformation> compact_streams(
      gdf_size_type num_columns, size_t num_rows, size_t num_data_streams,
      const std::vector<uint32_t>& stripe_list,
      hostdevice_vector<gpu::EncChunk>& chunks,
      hostdevice_vector<gpu::StripeStream>& strm_desc);

 private:
  compression_type compression_ = compression_type::none;
  bool enable_dictionary_ = true;

  std::vector<uint8_t> buffer_;
  std::ofstream outfile_;
};

}  // namespace orc
}  // namespace io
}  // namespace cudf
