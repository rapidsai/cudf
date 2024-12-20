/*
 * Copyright (c) 2020-2024, NVIDIA CORPORATION.
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

#include "text/subword/detail/codepoint_metadata.ah"
#include "text/subword/detail/tokenizer_utils.cuh"

#include <cudf/column/column_factories.hpp>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/strings/detail/utilities.cuh>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <nvtext/detail/load_hash_file.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

#include <thrust/fill.h>

#include <algorithm>
#include <cstdint>
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
rmm::device_uvector<codepoint_metadata_type> get_codepoint_metadata(rmm::cuda_stream_view stream)
{
  auto table_vector = rmm::device_uvector<codepoint_metadata_type>(codepoint_metadata_size, stream);
  auto table        = table_vector.data();
  thrust::fill(rmm::exec_policy(stream),
               table + cp_section1_end,
               table + codepoint_metadata_size,
               codepoint_metadata_default_value);
  CUDF_CUDA_TRY(cudaMemcpyAsync(table,
                                codepoint_metadata,
                                cp_section1_end * sizeof(codepoint_metadata[0]),  // 1st section
                                cudaMemcpyDefault,
                                stream.value()));
  CUDF_CUDA_TRY(cudaMemcpyAsync(
    table + cp_section2_begin,
    cp_metadata_917505_917999,
    (cp_section2_end - cp_section2_begin + 1) * sizeof(codepoint_metadata[0]),  // 2nd section
    cudaMemcpyDefault,
    stream.value()));
  return table_vector;
}

/**
 * @brief Retrieve the aux code point data table.
 *
 * Build the aux code point data table in device memory
 * using the vector pieces from codepoint_metadata.ah
 */
rmm::device_uvector<aux_codepoint_data_type> get_aux_codepoint_data(rmm::cuda_stream_view stream)
{
  auto table_vector = rmm::device_uvector<aux_codepoint_data_type>(aux_codepoint_data_size, stream);
  auto table        = table_vector.data();
  thrust::fill(rmm::exec_policy(stream),
               table + aux_section1_end,
               table + aux_codepoint_data_size,
               aux_codepoint_default_value);
  CUDF_CUDA_TRY(cudaMemcpyAsync(table,
                                aux_codepoint_data,
                                aux_section1_end * sizeof(aux_codepoint_data[0]),  // 1st section
                                cudaMemcpyDefault,
                                stream.value()));
  CUDF_CUDA_TRY(cudaMemcpyAsync(
    table + aux_section2_begin,
    aux_cp_data_44032_55203,
    (aux_section2_end - aux_section2_begin + 1) * sizeof(aux_codepoint_data[0]),  // 2nd section
    cudaMemcpyDefault,
    stream.value()));
  CUDF_CUDA_TRY(cudaMemcpyAsync(
    table + aux_section3_begin,
    aux_cp_data_70475_71099,
    (aux_section3_end - aux_section3_begin + 1) * sizeof(aux_codepoint_data[0]),  // 3rd section
    cudaMemcpyDefault,
    stream.value()));
  CUDF_CUDA_TRY(cudaMemcpyAsync(
    table + aux_section4_begin,
    aux_cp_data_119134_119232,
    (aux_section4_end - aux_section4_begin + 1) * sizeof(aux_codepoint_data[0]),  // 4th section
    cudaMemcpyDefault,
    stream.value()));
  return table_vector;
}

namespace {
/**
 * @brief Convert string to uint32.
 *
 * This just wraps the std::stoi but provides a nice error message
 * in case the hash file format is incorrect.
 */
uint32_t str_to_uint32(std::string const& str, uint64_t line_no)
{
  try {
    return std::stoi(str);  // there is no std::stoui
  } catch (std::exception const& exc) {
    std::string message("Line ");
    message += std::to_string(line_no) + ": ";
    message += "cannot convert integer from '";
    message += str;
    message += "': ";
    message += exc.what();
    std::cerr << message << std::endl;
    throw;
  }
}

/**
 * @brief Convert string to uint64.
 *
 * This just wraps the std::stoul but provides a nice error message
 * in case the hash file format is incorrect.
 */
uint64_t str_to_uint64(std::string const& str, uint64_t line_no)
{
  try {
    return std::stoul(str);
  } catch (std::exception const& exc) {
    std::string message("Line ");
    message += std::to_string(line_no) + ": ";
    message += "cannot convert integer from '";
    message += str;
    message += "': ";
    message += exc.what();
    std::cerr << message << std::endl;
    throw;
  }
}
}  // namespace

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
std::unique_ptr<hashed_vocabulary> load_vocabulary_file(
  std::string const& filename_hashed_vocabulary,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr)
{
  hashed_vocabulary result;
  std::ifstream hash_file(filename_hashed_vocabulary);
  CUDF_EXPECTS(hash_file.good(), "Could not open " + filename_hashed_vocabulary);

  uint64_t line_no = 1;
  std::string line;
  std::getline(hash_file, line);
  result.outer_hash_a = str_to_uint32(line, line_no++);

  std::getline(hash_file, line);
  result.outer_hash_b = str_to_uint32(line, line_no++);

  std::getline(hash_file, line);
  result.num_bins = str_to_uint32(line, line_no++);

  std::vector<uint64_t> bin_coefficients(result.num_bins);
  std::vector<uint16_t> bin_offsets(result.num_bins);

  for (int i = 0; i < result.num_bins; ++i) {
    std::getline(hash_file, line);
    size_t loc_of_space = line.find(" ");
    CUDF_EXPECTS(loc_of_space != line.npos, "invalid hash file format");

    std::string first_num  = line.substr(0, loc_of_space);
    std::string second_num = line.substr(loc_of_space + 1, line.length());

    bin_coefficients[i] = str_to_uint64(first_num, line_no);
    bin_offsets[i]      = str_to_uint32(second_num, line_no);
    ++line_no;
  }

  std::getline(hash_file, line);
  uint64_t hash_table_length = str_to_uint64(line, line_no++);
  std::vector<uint64_t> table(hash_table_length);

  std::generate(table.begin(), table.end(), [&hash_file, &line_no]() {
    std::string line;
    std::getline(hash_file, line);
    return str_to_uint64(line, line_no++);
  });

  std::getline(hash_file, line);
  result.unknown_token_id = str_to_uint32(line, line_no++);

  std::getline(hash_file, line);
  result.first_token_id = str_to_uint32(line, line_no++);

  std::getline(hash_file, line);
  result.separator_token_id = str_to_uint32(line, line_no++);

  // Transfer hash table to columns
  result.table = cudf::make_numeric_column(cudf::data_type{cudf::type_id::UINT64},
                                           table.size(),
                                           cudf::mask_state::UNALLOCATED,
                                           stream,
                                           mr);
  CUDF_CUDA_TRY(cudaMemcpyAsync(result.table->mutable_view().data<uint64_t>(),
                                table.data(),
                                table.size() * sizeof(uint64_t),
                                cudaMemcpyDefault,
                                stream.value()));

  result.bin_coefficients = cudf::make_numeric_column(cudf::data_type{cudf::type_id::UINT64},
                                                      bin_coefficients.size(),
                                                      cudf::mask_state::UNALLOCATED,
                                                      stream,
                                                      mr);
  CUDF_CUDA_TRY(cudaMemcpyAsync(result.bin_coefficients->mutable_view().data<uint64_t>(),
                                bin_coefficients.data(),
                                bin_coefficients.size() * sizeof(uint64_t),
                                cudaMemcpyDefault,
                                stream.value()));

  result.bin_offsets = cudf::make_numeric_column(cudf::data_type{cudf::type_id::UINT16},
                                                 bin_offsets.size(),
                                                 cudf::mask_state::UNALLOCATED,
                                                 stream,
                                                 mr);
  CUDF_CUDA_TRY(cudaMemcpyAsync(result.bin_offsets->mutable_view().data<uint16_t>(),
                                bin_offsets.data(),
                                bin_offsets.size() * sizeof(uint16_t),
                                cudaMemcpyDefault,
                                stream.value()));

  auto cp_metadata            = detail::get_codepoint_metadata(stream);
  auto const cp_metadata_size = static_cast<cudf::size_type>(cp_metadata.size());
  result.cp_metadata = std::make_unique<cudf::column>(cudf::data_type{cudf::type_id::UINT32},
                                                      cp_metadata_size,
                                                      cp_metadata.release(),
                                                      rmm::device_buffer{},
                                                      0);

  auto aux_cp_table            = detail::get_aux_codepoint_data(stream);
  auto const aux_cp_table_size = static_cast<cudf::size_type>(aux_cp_table.size());
  result.aux_cp_table = std::make_unique<cudf::column>(cudf::data_type{cudf::type_id::UINT64},
                                                       aux_cp_table_size,
                                                       aux_cp_table.release(),
                                                       rmm::device_buffer{},
                                                       0);

  return std::make_unique<hashed_vocabulary>(std::move(result));
}

}  // namespace detail

std::unique_ptr<hashed_vocabulary> load_vocabulary_file(
  std::string const& filename_hashed_vocabulary,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  return detail::load_vocabulary_file(filename_hashed_vocabulary, stream, mr);
}

}  // namespace nvtext
