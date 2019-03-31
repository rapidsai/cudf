/*
 * Copyright 2019 BlazingDB, Inc.
 *     Copyright 2019 Christian Noboa Mardini <christian@blazingdb.com>
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
#include <random>
#include <algorithm>
#include "types.hpp"
#include "tests/utilities/column_wrapper.cuh"

constexpr gdf_size_type INPUT_SIZE{107};
constexpr gdf_size_type BITSET_SIZE{128};


template <typename ColumnType, typename = void>
struct VectorRandomGenerator;

template <typename ColumnType>
struct VectorRandomGenerator
<
  ColumnType,
  typename std::enable_if_t<std::is_integral<ColumnType>::value, void>
> {
  ColumnType operator()() {
    return generator(random_device);
  }

  std::random_device random_device;
  std::uniform_int_distribution<ColumnType> generator;
};

template <typename ColumnType>
struct VectorRandomGenerator
<
  ColumnType,
  typename std::enable_if_t<std::is_floating_point<ColumnType>::value, void>
> {
  ColumnType operator()() {
    return generator(random_device);
  }

  std::random_device random_device;
  std::uniform_real_distribution<ColumnType> generator;
};


template <typename ColumnType>
cudf::test::column_wrapper<ColumnType> create_random_column(gdf_size_type size) {
  std::vector<ColumnType> data(size);
  VectorRandomGenerator<ColumnType> data_random_generator;
  std::generate_n(data.begin(), size, std::ref(data_random_generator));

  gdf_size_type bitmask_size = gdf_valid_allocation_size(size);
  std::vector<gdf_valid_type> bitmask(bitmask_size);
  VectorRandomGenerator<gdf_valid_type> bitmask_random_generator;
  std::generate_n(bitmask.begin(), bitmask_size, std::ref(bitmask_random_generator));

  return cudf::test::column_wrapper<ColumnType>(std::move(data), std::move(bitmask));
}


template <typename ColumnType>
std::vector<gdf_column*> allocate_slice_output_columns(
      std::vector<std::shared_ptr<cudf::test::column_wrapper<ColumnType>>>& output_columns,
      std::vector<gdf_index_type>& indexes) {
  for (std::size_t i = 0; i < indexes.size(); i += 2) {
    gdf_size_type size = indexes[i + 1] - indexes[i];
    output_columns.emplace_back(
        std::make_shared<cudf::test::column_wrapper<ColumnType>>(size, true));
  }

  std::vector<gdf_column*> source_columns;
  std::transform(output_columns.begin(),
                 output_columns.end(),
                 std::back_inserter(source_columns),
                 [](std::shared_ptr<cudf::test::column_wrapper<ColumnType>>& column){
                   return column->get();  
                 });

  return source_columns;
}


template <typename ColumnType>
std::vector<gdf_column*> allocate_split_output_columns(
      std::vector<std::shared_ptr<cudf::test::column_wrapper<ColumnType>>>& output_columns,
      std::vector<gdf_index_type>& indexes,
      gdf_size_type input_size) {
  gdf_size_type init_index{0};
  for (std::size_t i = 0; i <= indexes.size(); ++i) {
    gdf_size_type size = input_size - init_index;
    if (i < indexes.size()) {
      size = indexes[i] - init_index;
      init_index = indexes[i];
    }
    output_columns.emplace_back(
        std::make_shared<cudf::test::column_wrapper<ColumnType>>(size, true));
  }

  std::vector<gdf_column*> source_columns;
  std::transform(output_columns.begin(),
                 output_columns.end(),
                 std::back_inserter(source_columns),
                 [](std::shared_ptr<cudf::test::column_wrapper<ColumnType>>& column){
                   return column->get();  
                 });

  return source_columns;
}


template <typename ColumnType>
struct HelperColumn {
  gdf_size_type bit_set_count;
  std::vector<ColumnType> data;
  std::vector<gdf_valid_type> bitmask;
};


template <typename ColumnType>
HelperColumn<ColumnType> makeHelperColumn(
    cudf::test::column_wrapper<ColumnType>& column) {
  auto column_host = column.to_host();

  HelperColumn<ColumnType> result;
  result.bit_set_count = column.get()->null_count;
  result.data = std::get<0>(column_host);
  result.bitmask = std::get<1>(column_host);

  return result;
}


template <typename ColumnType>
std::vector<HelperColumn<ColumnType>> makeHelperColumn(
    std::vector<std::shared_ptr<cudf::test::column_wrapper<ColumnType>>>& columns) {
  std::vector<HelperColumn<ColumnType>> result;
  for (auto& column_wrapper : columns) {
    auto column = column_wrapper->to_host();
    result.emplace_back(
        HelperColumn<ColumnType>{.bit_set_count = column_wrapper->get()->null_count,
                                 .data = std::get<0>(column),
                                 .bitmask = std::get<1>(column)});
  }
  return result;
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
std::vector<HelperColumn<ColumnType>> slice_columns(
      HelperColumn<ColumnType>& input_column,
      std::vector<gdf_index_type>& indexes) {
  std::vector<HelperColumn<ColumnType>> output_column_cpu;
  for (std::size_t i = 0; i < indexes.size(); i += 2) {
    gdf_size_type init_index = indexes[i];
    gdf_size_type end_index = indexes[i + 1];

    if (init_index == end_index) {
      HelperColumn<ColumnType> column {.bit_set_count = 0,
                                       .data = std::vector<ColumnType>(),
                                       .bitmask = std::vector<gdf_valid_type>() };
      output_column_cpu.emplace_back(column);
      continue;
    }

    HelperColumn<ColumnType> helper_column;
    helper_column.data =
        std::vector<ColumnType>(input_column.data.begin() + init_index,
                                input_column.data.begin() + end_index);
    helper_column.bitmask =
        slice_cpu_valids<BITSET_SIZE>(helper_column.bit_set_count,
                                      input_column.bitmask,
                                      init_index,
                                      end_index);

    output_column_cpu.emplace_back(helper_column);
  }
  return output_column_cpu;
}


template <typename ColumnType>
std::vector<HelperColumn<ColumnType>> split_columns(
      HelperColumn<ColumnType>& input_column,
      std::vector<gdf_index_type>& indexes,
      gdf_size_type const input_size) {
  std::vector<HelperColumn<ColumnType>> output_column_cpu;
  for (std::size_t i = 0; i <= indexes.size(); ++i) {
    gdf_size_type init_index{0};
    if (i != 0) {
      init_index = indexes[i - 1];
    }
    gdf_size_type end_index = input_size;
    if (i < indexes.size()) {
      end_index = indexes[i];
    }

    if (init_index == end_index) {
      HelperColumn<ColumnType> column {.bit_set_count = 0,
                                       .data = std::vector<ColumnType>(),
                                       .bitmask = std::vector<gdf_valid_type>() };
      output_column_cpu.emplace_back(column);
      continue;
    }

    HelperColumn<ColumnType> helper_column;
    helper_column.data =
        std::vector<ColumnType>(input_column.data.begin() + init_index,
                                input_column.data.begin() + end_index);
    helper_column.bitmask =
        slice_cpu_valids<BITSET_SIZE>(helper_column.bit_set_count,
                                      input_column.bitmask,
                                      init_index,
                                      end_index);

    output_column_cpu.emplace_back(helper_column);
  }
  return output_column_cpu;
}


template <typename Type, template <typename> typename Column = HelperColumn>
void verify(HelperColumn<Type> const& lhs, HelperColumn<Type> const& rhs) {
  // Compare null count
  ASSERT_EQ(lhs.bit_set_count, rhs.bit_set_count);

  // Compare data
  ASSERT_EQ(lhs.data.size(), rhs.data.size());
  for (gdf_size_type i = 0; i < (gdf_size_type)lhs.data.size(); ++i) {
    ASSERT_EQ(lhs.data[i], rhs.data[i]);
  }

  // Compare bitmask
  ASSERT_EQ(lhs.bitmask.size(), rhs.bitmask.size());
  for (gdf_size_type i = 0; i < (gdf_size_type)lhs.bitmask.size(); ++i) {
    ASSERT_EQ(lhs.bitmask[i], rhs.bitmask[i]);
  }
}

#endif
