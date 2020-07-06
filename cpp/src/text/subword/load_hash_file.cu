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

#include <cudf/column/column_factories.hpp>
#include <cudf/utilities/error.hpp>
#include <nvtext/detail/load_hash_file.hpp>

#include <stdint.h>
#include <algorithm>
#include <fstream>
#include <iostream>
#include <vector>

namespace nvtext {
namespace detail {

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
                                       cudaStream_t stream,
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
                           stream));

  result.bin_coefficients = cudf::make_numeric_column(cudf::data_type{cudf::type_id::UINT64},
                                                      bin_coefficients.size(),
                                                      cudf::mask_state::UNALLOCATED,
                                                      stream,
                                                      mr);
  CUDA_TRY(cudaMemcpyAsync(result.bin_coefficients->mutable_view().data<uint64_t>(),
                           bin_coefficients.data(),
                           bin_coefficients.size() * sizeof(uint64_t),
                           cudaMemcpyHostToDevice,
                           stream));

  result.bin_offsets = cudf::make_numeric_column(cudf::data_type{cudf::type_id::UINT16},
                                                 bin_offsets.size(),
                                                 cudf::mask_state::UNALLOCATED,
                                                 stream,
                                                 mr);
  CUDA_TRY(cudaMemcpyAsync(result.bin_offsets->mutable_view().data<uint16_t>(),
                           bin_offsets.data(),
                           bin_offsets.size() * sizeof(uint16_t),
                           cudaMemcpyHostToDevice,
                           stream));

  return result;
}

}  // namespace detail

hashed_vocabulary load_vocabulary_file(std::string const& filename_hashed_vocabulary,
                                       rmm::mr::device_memory_resource* mr)
{
  return detail::load_vocabulary_file(filename_hashed_vocabulary, 0, mr);
}

}  // namespace nvtext
