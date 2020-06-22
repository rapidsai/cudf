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

#include <rmm/thrust_rmm_allocator.h>
#include <stdint.h>
#include <thrust/device_vector.h>
#include <fstream>
#include <iostream>
#include <vector>

namespace nvtext {
namespace detail {

void load_hash_information(const std::string& path,
                           uint32_t& outer_table_a,
                           uint32_t& outer_table_b,
                           uint16_t& num_bins,
                           uint16_t& unk_tok_id,
                           uint16_t& first_tok_id,
                           uint16_t& sep_tok_id,
                           std::vector<uint64_t>& hash_table,
                           std::vector<uint64_t>& bin_coefficients,
                           std::vector<uint16_t>& bin_offsets)
{
  std::ifstream hash_file(path);
  if (!hash_file.good()) {
    std::cerr << "Hash file " << path << " not found." << std::endl;
    exit(1);
  }

  std::string line;
  std::getline(hash_file, line);
  outer_table_a = std::stoi(line);

  std::getline(hash_file, line);
  outer_table_b = std::stoi(line);

  std::getline(hash_file, line);
  num_bins = std::stoi(line);

  bin_coefficients.resize(num_bins);
  bin_offsets.resize(num_bins);

  for (int i = 0; i < num_bins; ++i) {
    std::getline(hash_file, line);
    size_t loc_of_space = line.find(" ");

    std::string first_num  = line.substr(0, loc_of_space);
    std::string second_num = line.substr(loc_of_space + 1, line.length());

    bin_coefficients[i] = std::stoull(first_num);
    bin_offsets[i]      = std::stoull(second_num);
  }

  std::getline(hash_file, line);
  uint64_t hash_table_length = std::stoull(line);
  hash_table.resize(hash_table_length);

  for (uint32_t i = 0; i < hash_table_length; ++i) {
    std::getline(hash_file, line);
    hash_table[i] = std::stoull(line);
  }

  std::getline(hash_file, line);
  unk_tok_id = std::stoi(line);

  std::getline(hash_file, line);
  first_tok_id = std::stoi(line);

  std::getline(hash_file, line);
  sep_tok_id = std::stoi(line);
}

void transfer_hash_info_to_device(const std::string hash_data_file,
                                  rmm::device_vector<uint64_t>& device_hash_table,
                                  rmm::device_vector<uint64_t>& device_bin_coefficients,
                                  rmm::device_vector<uint16_t>& device_bin_offsets,
                                  uint16_t& unk_tok_id,
                                  uint16_t& first_tok_id,
                                  uint16_t& sep_tok_id,
                                  uint32_t& outer_table_a,
                                  uint32_t& outer_table_b,
                                  uint16_t& num_bins)
{
  std::vector<uint64_t> hash_table;
  std::vector<uint64_t> bin_coefficients;
  std::vector<uint16_t> bin_offsets;

  load_hash_information(hash_data_file,
                        outer_table_a,
                        outer_table_b,
                        num_bins,
                        unk_tok_id,
                        first_tok_id,
                        sep_tok_id,
                        hash_table,
                        bin_coefficients,
                        bin_offsets);

  // Transfer hash table vectors
  device_hash_table       = hash_table;
  device_bin_coefficients = bin_coefficients;
  device_bin_offsets      = bin_offsets;
}

}  // namespace detail
}  // namespace nvtext
