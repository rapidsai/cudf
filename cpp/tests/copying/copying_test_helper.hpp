/*
 * Copyright 2019 BlazingDB, Inc.
 *     Copyright 2019 Christian Noboa Mardini <christian@blazingdb.com>
 *     Copyright 2019 William Scott Malpica <william@blazingdb.com>
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

#ifndef TESTS_COPYING_COPYINGTESTHELPER_HPP
#define TESTS_COPYING_COPYINGTESTHELPER_HPP

#include <vector>
#include <cstdlib>
#include <algorithm>
#include <cudf/types.hpp>
#include <tests/utilities/column_wrapper.cuh>

constexpr gdf_size_type INPUT_SIZE{107};
constexpr gdf_size_type BITSET_SIZE{128};

template <typename ColumnType>
cudf::test::column_wrapper<ColumnType> create_random_column(gdf_size_type size) {
  
  return cudf::test::column_wrapper<ColumnType>(size, 
            [](gdf_index_type row) { return static_cast <ColumnType>(rand()); }, 
            [](gdf_index_type row) { return (rand() % 3 == 0) ? false : true; });
}


template <gdf_size_type SIZE>
std::vector<gdf_valid_type> slice_cpu_valids(gdf_size_type& bit_set_counter,
                                             std::vector<gdf_valid_type> const& input_valid,
                                             gdf_size_type index_start,
                                             gdf_size_type index_final) {
  // delta of the indexes
  gdf_size_type bits_length = index_final - index_start;
  if (bits_length == 0) {
    return std::vector<gdf_valid_type>();
  }

  // bitset data
  std::bitset<SIZE> bitset{};

  // populate data into bitset
  gdf_size_type bit_index{0};
  auto put_data_bitset = [&bitset, &bit_index](gdf_valid_type value) {
    for (int k = 0; k < 8; ++k) {
      if (SIZE <= bit_index) {
          break;
      }
      bitset[bit_index] = value & 1;
      value >>= 1;
      bit_index++;
    }
  };
  std::for_each(input_valid.begin(), input_valid.end(), put_data_bitset);

  // perform shift operation
  bitset >>= index_start;

  // calculate result byte size with padding
  gdf_size_type result_byte_size_padding = gdf_valid_allocation_size(bits_length);

  // extract data from bitset
  bit_index = 0;
  bit_set_counter = 0;
  auto get_data_bitset = [&bitset, &bit_index, &bits_length, &bit_set_counter]() {
    gdf_valid_type value = 0;
    for (int k = 0; k < 8; ++k) {
      if (bits_length <= bit_index) {
          return value;
      }
      gdf_valid_type tmp = bitset[bit_index];
      value |= (tmp << k);
      bit_index++;
      if ((tmp & 1) == 0) {
        bit_set_counter++;
      }
    }
    return value;
  };

  // create and store bitset into result
  std::vector<gdf_valid_type> result(result_byte_size_padding, 0);
  std::generate_n(result.begin(), result_byte_size_padding, get_data_bitset);

  // done
  return result;
}


template <typename ColumnType>
auto slice_columns(
      std::vector<ColumnType>& input_col_data,
      std::vector<gdf_valid_type>& input_col_bitmask,
      std::vector<gdf_index_type>& indexes) {
  
  std::vector<std::vector<ColumnType>> output_cols_data;
  std::vector<std::vector<gdf_valid_type>> output_cols_bitmask;
  std::vector<gdf_size_type> output_cols_null_count;

  for (std::size_t i = 0; i < indexes.size(); i += 2) {
    gdf_size_type init_index = indexes[i];
    gdf_size_type end_index = indexes[i + 1];

    if (init_index == end_index) {
      output_cols_data.emplace_back(std::vector<ColumnType>());
      output_cols_bitmask.emplace_back(std::vector<gdf_valid_type>());
      output_cols_null_count.emplace_back(0);
    } else {

      output_cols_data.emplace_back(
          std::vector<ColumnType>(input_col_data.begin() + init_index,
                                  input_col_data.begin() + end_index));

      gdf_size_type bit_set_counter=0;
      output_cols_bitmask.emplace_back(
        slice_cpu_valids<BITSET_SIZE>(bit_set_counter,
                                        input_col_bitmask,
                                        init_index,
                                        end_index));
      output_cols_null_count.emplace_back(bit_set_counter);      
    }
  }
  return std::make_tuple(output_cols_data, output_cols_bitmask, output_cols_null_count);
}


template <typename ColumnType>
auto split_columns(
      std::vector<ColumnType>& input_col_data,
      std::vector<gdf_valid_type>& input_col_bitmask,
      std::vector<gdf_index_type>& indexes) {

  std::vector<std::vector<ColumnType>> output_cols_data;
  std::vector<std::vector<gdf_valid_type>> output_cols_bitmask;
  std::vector<gdf_size_type> output_cols_null_count;

  for (std::size_t i = 0; i <= indexes.size(); ++i) {
    gdf_size_type init_index{0};
    if (i != 0) {
      init_index = indexes[i - 1];
    }
    gdf_size_type end_index = input_col_data.size();
    if (i < indexes.size()) {
      end_index = indexes[i];
    }

    if (init_index == end_index) {
      output_cols_data.emplace_back(std::vector<ColumnType>());
      output_cols_bitmask.emplace_back(std::vector<gdf_valid_type>());
      output_cols_null_count.emplace_back(0);
    } else {
      output_cols_data.emplace_back(
          std::vector<ColumnType>(input_col_data.begin() + init_index,
                                  input_col_data.begin() + end_index));

      gdf_size_type bit_set_counter=0;
      output_cols_bitmask.emplace_back(
        slice_cpu_valids<BITSET_SIZE>(bit_set_counter,
                                        input_col_bitmask,
                                        init_index,
                                        end_index));
      output_cols_null_count.emplace_back(bit_set_counter);   
    }       
  }
  return std::make_tuple(output_cols_data, output_cols_bitmask, output_cols_null_count);
}

#endif
