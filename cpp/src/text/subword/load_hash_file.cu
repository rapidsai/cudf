/*
 * Copyright (c) 2020, NVIDIA CORPORATION.
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

#include <text/subword/detail/codepoint_metadata.ah>
#include <text/subword/detail/data_normalizer.hpp>
#include <text/subword/detail/tokenizer_utils.cuh>

#include <nvtext/detail/load_hash_file.hpp>

#include <cudf/column/column_factories.hpp>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/strings/detail/utilities.cuh>
#include <cudf/utilities/error.hpp>

#include <rmm/cuda_stream_view.hpp>

#include <stdint.h>
#include <algorithm>
#include <fstream>
#include <iostream>
#include <vector>

namespace nvtext {
namespace detail {

/**
 * @brief Retrieve the code point metadata table.
 *
 * Build the code point metadata table in device memory
 * using the vector pieces from codepoint_metadata.ah
 */
const codepoint_metadata_type* get_codepoint_metadata(rmm::cuda_stream_view stream)
{
  static cudf::strings::detail::thread_safe_per_context_cache<codepoint_metadata_type>
    g_codepoint_metadata;
  return g_codepoint_metadata.find_or_initialize([stream](void) {
    codepoint_metadata_type* table =
      static_cast<codepoint_metadata_type*>(rmm::mr::get_current_device_resource()->allocate(
        codepoint_metadata_size * sizeof(codepoint_metadata_type), stream));
    thrust::fill(rmm::exec_policy(stream)->on(stream.value()),
                 table + cp_section1_end,
                 table + codepoint_metadata_size,
                 codepoint_metadata_default_value);
    CUDA_TRY(cudaMemcpyAsync(table,
                             codepoint_metadata,
                             cp_section1_end * sizeof(codepoint_metadata[0]),  // 1st section
                             cudaMemcpyHostToDevice,
                             stream.value()));
    CUDA_TRY(cudaMemcpyAsync(
      table + cp_section2_begin,
      cp_metadata_917505_917999,
      (cp_section2_end - cp_section2_begin + 1) * sizeof(codepoint_metadata[0]),  // 2nd section
      cudaMemcpyHostToDevice,
      stream.value()));
    return table;
  });
}

/**
 * @brief Retrieve the aux code point data table.
 *
 * Build the aux code point data table in device memory
 * using the vector pieces from codepoint_metadata.ah
 */
const aux_codepoint_data_type* get_aux_codepoint_data(rmm::cuda_stream_view stream)
{
  static cudf::strings::detail::thread_safe_per_context_cache<aux_codepoint_data_type>
    g_aux_codepoint_data;
  return g_aux_codepoint_data.find_or_initialize([stream](void) {
    aux_codepoint_data_type* table =
      static_cast<aux_codepoint_data_type*>(rmm::mr::get_current_device_resource()->allocate(
        aux_codepoint_data_size * sizeof(aux_codepoint_data_type), stream));
    thrust::fill(rmm::exec_policy(stream)->on(stream.value()),
                 table + aux_section1_end,
                 table + aux_codepoint_data_size,
                 aux_codepoint_default_value);
    CUDA_TRY(cudaMemcpyAsync(table,
                             aux_codepoint_data,
                             aux_section1_end * sizeof(aux_codepoint_data[0]),  // 1st section
                             cudaMemcpyHostToDevice,
                             stream.value()));
    CUDA_TRY(cudaMemcpyAsync(
      table + aux_section2_begin,
      aux_cp_data_44032_55203,
      (aux_section2_end - aux_section2_begin + 1) * sizeof(aux_codepoint_data[0]),  // 2nd section
      cudaMemcpyHostToDevice,
      stream.value()));
    CUDA_TRY(cudaMemcpyAsync(
      table + aux_section3_begin,
      aux_cp_data_70475_71099,
      (aux_section3_end - aux_section3_begin + 1) * sizeof(aux_codepoint_data[0]),  // 3rd section
      cudaMemcpyHostToDevice,
      stream.value()));
    CUDA_TRY(cudaMemcpyAsync(
      table + aux_section4_begin,
      aux_cp_data_119134_119232,
      (aux_section4_end - aux_section4_begin + 1) * sizeof(aux_codepoint_data[0]),  // 4th section
      cudaMemcpyHostToDevice,
      stream.value()));
    return table;
  });
}

/**
 * @brief Loads a text file representing the hashed vocabulary into hashed_vocabulary struct.
 *
 * @code{.pseudo}
 * Format of the file (ASCII text file with numbers):
 * First 3 lines have the following values:
 *  outer_hash_a
 *  outer_hash_b
 *  number-of-bins
 * The next number-of-bins lines has two values in each line separated by a space
 *  coefficient offset
 *  ...
 * Next line has the size (number of lines) of the table followed
 * by the table values -- one value per line.
 * The last three lines:
 *  unknown_token_id
 *  first_token_id
 *  separator_token_id
 * @endcode
 *
 * @param filename_hashed_vocabulary Path to text file containing hashed vocabulary
 * @return object containing hash table elements for the wordpiece tokenizer
 */
hashed_vocabulary load_vocabulary_file(std::string const& filename_hashed_vocabulary,
                                       rmm::cuda_stream_view stream,
                                       rmm::mr::device_memory_resource* mr)
{
  hashed_vocabulary result;
  std::ifstream hash_file(filename_hashed_vocabulary);
  CUDF_EXPECTS(hash_file.good(), "Could not open " + filename_hashed_vocabulary);

  std::string line;
  std::getline(hash_file, line);
  result.outer_hash_a = std::stoi(line);

  std::getline(hash_file, line);
  result.outer_hash_b = std::stoi(line);

  std::getline(hash_file, line);
  result.num_bins = std::stoi(line);

  std::vector<uint64_t> bin_coefficients(result.num_bins);
  std::vector<uint16_t> bin_offsets(result.num_bins);

  for (int i = 0; i < result.num_bins; ++i) {
    std::getline(hash_file, line);
    size_t loc_of_space = line.find(" ");

    std::string first_num  = line.substr(0, loc_of_space);
    std::string second_num = line.substr(loc_of_space + 1, line.length());

    bin_coefficients[i] = std::stoull(first_num);
    bin_offsets[i]      = std::stoull(second_num);
  }

  std::getline(hash_file, line);
  uint64_t hash_table_length = std::stoull(line);
  std::vector<uint64_t> table(hash_table_length);

  std::generate(table.begin(), table.end(), [&hash_file]() {
    std::string line;
    std::getline(hash_file, line);
    return std::stoull(line);
  });

  std::getline(hash_file, line);
  result.unknown_token_id = std::stoi(line);

  std::getline(hash_file, line);
  result.first_token_id = std::stoi(line);

  std::getline(hash_file, line);
  result.separator_token_id = std::stoi(line);

  // Transfer hash table to columns
  result.table = cudf::make_numeric_column(cudf::data_type{cudf::type_id::UINT64},
                                           table.size(),
                                           cudf::mask_state::UNALLOCATED,
                                           stream,
                                           mr);
  CUDA_TRY(cudaMemcpyAsync(result.table->mutable_view().data<uint64_t>(),
                           table.data(),
                           table.size() * sizeof(uint64_t),
                           cudaMemcpyHostToDevice,
                           stream.value()));

  result.bin_coefficients = cudf::make_numeric_column(cudf::data_type{cudf::type_id::UINT64},
                                                      bin_coefficients.size(),
                                                      cudf::mask_state::UNALLOCATED,
                                                      stream,
                                                      mr);
  CUDA_TRY(cudaMemcpyAsync(result.bin_coefficients->mutable_view().data<uint64_t>(),
                           bin_coefficients.data(),
                           bin_coefficients.size() * sizeof(uint64_t),
                           cudaMemcpyHostToDevice,
                           stream.value()));

  result.bin_offsets = cudf::make_numeric_column(cudf::data_type{cudf::type_id::UINT16},
                                                 bin_offsets.size(),
                                                 cudf::mask_state::UNALLOCATED,
                                                 stream,
                                                 mr);
  CUDA_TRY(cudaMemcpyAsync(result.bin_offsets->mutable_view().data<uint16_t>(),
                           bin_offsets.data(),
                           bin_offsets.size() * sizeof(uint16_t),
                           cudaMemcpyHostToDevice,
                           stream.value()));

  // this just initializes some constant tables into device memory
  // to help speed up the runtime
  detail::get_codepoint_metadata(stream);
  detail::get_aux_codepoint_data(stream);

  return result;
}

}  // namespace detail

hashed_vocabulary load_vocabulary_file(std::string const& filename_hashed_vocabulary,
                                       rmm::mr::device_memory_resource* mr)
{
  CUDF_FUNC_RANGE();
  return detail::load_vocabulary_file(filename_hashed_vocabulary, rmm::cuda_stream_default, mr);
}

}  // namespace nvtext
