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

#include <cudf/copying.hpp>
#include <cudf/legacy/table.hpp>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <tests/utilities/column_wrapper.cuh>
#include "tests/utilities/compare_column_wrappers.cuh"
#include <tests/utilities/cudf_test_fixtures.h>
#include <tests/utilities/cudf_test_utils.cuh>
#include <tests/utilities/compare_column_wrappers.cuh>
#include <cudf/types.hpp>
#include <cudf/utilities/legacy/wrapper_types.hpp>

#include <random>
#include <cudf/transform.hpp>

using bit_mask::bit_mask_t;

template <typename T>
struct bitmask_test : GdfTest {};

using test_types =
    ::testing::Types<float, double>;
TYPED_TEST_CASE(bitmask_test, test_types);

TYPED_TEST(bitmask_test, nans_to_nulls_source_mask_valid) {
  constexpr gdf_size_type source_size{2000};
  
  cudf::test::column_wrapper<TypeParam> source_column{
      source_size, [](gdf_index_type row) { return row % 3 ? static_cast<TypeParam>(row) : static_cast<TypeParam>(nan("")); },
      [](gdf_index_type row) { return row % 3 != 1; }};

  gdf_column* raw_source = source_column.get();

  const auto num_bytes = cudf::util::packed_bit_sequence_size_in_bytes<bit_mask_t>(source_size)*sizeof(bit_mask_t);
  std::vector<bit_mask_t> result_mask_host(num_bytes);

  std::pair<bit_mask_t*, gdf_size_type> result;
  EXPECT_NO_THROW(result = cudf::nans_to_nulls(*raw_source));
  
  EXPECT_EQ(result.second, 2*(source_size/3 + 1));

  CUDA_TRY(cudaMemcpy(result_mask_host.data(), result.first, num_bytes, cudaMemcpyDeviceToHost));
  
  for (gdf_index_type i = 0; i < source_size; i++) {
    // The first half of the destination column should be all valid
    // and values should be 1, 3, 5, 7, etc.
    if (i % 3 == 2) {
      EXPECT_TRUE(cudf::util::bit_is_set<bit_mask_t>(result_mask_host.data(), i))
          << "Value at index " << i << " should be non-null!\n";
    } else {
      EXPECT_TRUE(!cudf::util::bit_is_set<bit_mask_t>(result_mask_host.data(), i))
          << "Value at index " << i << " should be null!\n";
    }
  }

  RMM_TRY(RMM_FREE(result.first, 0));
}

TYPED_TEST(bitmask_test, nans_to_nulls_source_mask_null) {
  constexpr gdf_size_type source_size{2000};
  
  cudf::test::column_wrapper<TypeParam> source_column{
      source_size, [](gdf_index_type row) { return row % 3 ? static_cast<TypeParam>(row) : static_cast<TypeParam>(nan("")); }
  };

  gdf_column* raw_source = source_column.get();

  const auto num_bytes = cudf::util::packed_bit_sequence_size_in_bytes<bit_mask_t>(source_size)*sizeof(bit_mask_t);
  std::vector<bit_mask_t> result_mask_host(num_bytes);

  std::pair<bit_mask_t*, gdf_size_type> result;
  EXPECT_NO_THROW(result = cudf::nans_to_nulls(*raw_source));

  EXPECT_EQ(result.second, source_size/3 + 1);

  CUDA_TRY(cudaMemcpy(result_mask_host.data(), result.first, num_bytes, cudaMemcpyDeviceToHost));
  
  for (gdf_index_type i = 0; i < source_size; i++) {
    // The first half of the destination column should be all valid
    // and values should be 1, 3, 5, 7, etc.
    if (i % 3) {
      EXPECT_TRUE(cudf::util::bit_is_set<bit_mask_t>(result_mask_host.data(), i))
          << "Value at index " << i << " should be non-null!\n";
    } else {
      EXPECT_TRUE(!cudf::util::bit_is_set<bit_mask_t>(result_mask_host.data(), i))
          << "Value at index " << i << " should be null!\n";
    }
  }

  RMM_TRY(RMM_FREE(result.first, 0));
}

TYPED_TEST(bitmask_test, nans_to_nulls_source_empty) {
 
  gdf_column source{};

  std::pair<bit_mask_t*, gdf_size_type> result;

  EXPECT_NO_THROW(result = cudf::nans_to_nulls(source));

  EXPECT_EQ(result.second, 0);
  EXPECT_EQ(result.first, nullptr);

}

TEST_F(GdfTest, nans_to_nulls_wrong_type) {
  
  using TypeParam = int;

  constexpr gdf_size_type source_size{2000};
  
  cudf::test::column_wrapper<TypeParam> source_column{
      source_size, [](gdf_index_type row) { return row % 3 ? static_cast<TypeParam>(row) : static_cast<TypeParam>(nan("")); },
      [](gdf_index_type row) { return row % 3 != 1; }};

  gdf_column* raw_source = source_column.get();

  std::pair<bit_mask_t*, gdf_size_type> result;
  EXPECT_THROW(result = cudf::nans_to_nulls(*raw_source), cudf::logic_error);
  
}
