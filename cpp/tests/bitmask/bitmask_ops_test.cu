/*
 * Copyright (c) 2018, NVIDIA CORPORATION.
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

#include <tests/utilities/cudf_test_fixtures.h>
#include <bitmask/bit_mask.cuh>
#include <table/table.hpp>
#include <tests/utilities/column_wrapper.cuh>
#include <tests/utilities/cudf_test_utils.cuh>

#include <cudf.h>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <cuda_profiler_api.h>

#include <chrono>

struct ValidsTest : public GdfTest {};

TEST_F(ValidsTest, NoValids) {
  const int num_rows = 100;
  std::vector<int> data(num_rows);
  const int num_masks = std::ceil(num_rows / static_cast<float>(8));
  std::vector<gdf_valid_type> valid(num_masks, 0x00);

  auto input_gdf_col = create_gdf_column(data, valid);

  int count{-1};
  gdf_error error_code =
      gdf_count_nonzero_mask(input_gdf_col->valid, num_rows, &count);

  ASSERT_EQ(GDF_SUCCESS, error_code)
      << "GDF Operation did not complete successfully.";

  EXPECT_EQ(0, count);
}

TEST_F(ValidsTest, NullValids) {
  int count{-1};
  const gdf_size_type size{100};
  gdf_error error_code = gdf_count_nonzero_mask(nullptr, size, &count);

  ASSERT_EQ(GDF_SUCCESS, error_code)
      << "GDF Operation did not complete successfully.";
  EXPECT_EQ(size, count);
}

TEST_F(ValidsTest, NullCount) {
  std::vector<int> data(0);
  std::vector<gdf_valid_type> valid{0x0};
  auto input_gdf_col = create_gdf_column(data, valid);
  gdf_error error_code =
      gdf_count_nonzero_mask(input_gdf_col->valid, 1, nullptr);

  ASSERT_EQ(GDF_DATASET_EMPTY, error_code)
      << "Expected failure for null input.";
}

TEST_F(ValidsTest, FirstRowValid) {
  std::vector<int> data(4);
  std::vector<gdf_valid_type> valid{0x1};

  auto input_gdf_col = create_gdf_column(data, valid);

  int count{-1};
  gdf_error error_code =
      gdf_count_nonzero_mask(input_gdf_col->valid, 1, &count);

  ASSERT_EQ(GDF_SUCCESS, error_code)
      << "GDF Operation did not complete successfully.";

  EXPECT_EQ(1, count);
}

TEST_F(ValidsTest, EightRowsValid) {
  std::vector<int> data(8);
  std::vector<gdf_valid_type> valid{0xFF};

  auto input_gdf_col = create_gdf_column(data, valid);

  int count{-1};
  gdf_error error_code =
      gdf_count_nonzero_mask(input_gdf_col->valid, 8, &count);

  ASSERT_EQ(GDF_SUCCESS, error_code)
      << "GDF Operation did not complete successfully.";

  EXPECT_EQ(8, count);
}

TEST_F(ValidsTest, EveryOtherBit) {
  std::vector<int> data(8);
  std::vector<gdf_valid_type> valid{0xAA};

  auto input_gdf_col = create_gdf_column(data, valid);

  int count{-1};
  gdf_error error_code =
      gdf_count_nonzero_mask(input_gdf_col->valid, 8, &count);

  ASSERT_EQ(GDF_SUCCESS, error_code)
      << "GDF Operation did not complete successfully.";

  EXPECT_EQ(4, count);
}

TEST_F(ValidsTest, OtherEveryOtherBit) {
  std::vector<int> data(8);
  std::vector<gdf_valid_type> valid{0x55};

  auto input_gdf_col = create_gdf_column(data, valid);

  int count{-1};
  gdf_error error_code =
      gdf_count_nonzero_mask(input_gdf_col->valid, 8, &count);

  ASSERT_EQ(GDF_SUCCESS, error_code)
      << "GDF Operation did not complete successfully.";

  EXPECT_EQ(4, count);
}

TEST_F(ValidsTest, 15rows) {
  const int num_rows = 15;
  std::vector<int> data(num_rows);
  const int num_masks = std::ceil(num_rows / static_cast<float>(8));
  std::vector<gdf_valid_type> valid(num_masks, 0x01);

  auto input_gdf_col = create_gdf_column(data, valid);

  int count{-1};
  gdf_error error_code =
      gdf_count_nonzero_mask(input_gdf_col->valid, num_rows, &count);

  ASSERT_EQ(GDF_SUCCESS, error_code)
      << "GDF Operation did not complete successfully.";

  EXPECT_EQ(2, count);
}

TEST_F(ValidsTest, 5rows) {
  const int num_rows = 5;
  std::vector<int> data(num_rows);
  const int num_masks = std::ceil(num_rows / static_cast<float>(8));
  std::vector<gdf_valid_type> valid(num_masks, 0x01);

  auto input_gdf_col = create_gdf_column(data, valid);

  int count{-1};
  gdf_error error_code =
      gdf_count_nonzero_mask(input_gdf_col->valid, num_rows, &count);

  ASSERT_EQ(GDF_SUCCESS, error_code)
      << "GDF Operation did not complete successfully.";

  EXPECT_EQ(1, count);
}

TEST_F(ValidsTest, 10ValidRows) {
  const int num_rows = 10;
  std::vector<float> data(num_rows);
  const int num_masks = std::ceil(num_rows / static_cast<float>(8));
  std::vector<gdf_valid_type> valid(num_masks, 0xFF);

  auto input_gdf_col = create_gdf_column(data, valid);

  int count{-1};
  gdf_error error_code =
      gdf_count_nonzero_mask(input_gdf_col->valid, num_rows, &count);

  ASSERT_EQ(GDF_SUCCESS, error_code)
      << "GDF Operation did not complete successfully.";

  EXPECT_EQ(10, count);
}

TEST_F(ValidsTest, MultipleOfEight) {
  const int num_rows = 1024;
  std::vector<int> data(num_rows);

  const int num_masks = std::ceil(num_rows / static_cast<float>(8));
  std::vector<gdf_valid_type> valid(num_masks, 0x01);

  auto input_gdf_col = create_gdf_column(data, valid);

  int count{-1};
  gdf_error error_code =
      gdf_count_nonzero_mask(input_gdf_col->valid, num_rows, &count);

  ASSERT_EQ(GDF_SUCCESS, error_code)
      << "GDF Operation did not complete successfully.";

  EXPECT_EQ(128, count);
}

TEST_F(ValidsTest, NotMultipleOfEight) {
  const int num_rows = 1023;
  std::vector<int> data(num_rows);

  const int num_masks = std::ceil(num_rows / static_cast<float>(8));
  std::vector<gdf_valid_type> valid(num_masks, 0x80);

  auto input_gdf_col = create_gdf_column(data, valid);

  int count{-1};
  gdf_error error_code =
      gdf_count_nonzero_mask(input_gdf_col->valid, num_rows, &count);

  ASSERT_EQ(GDF_SUCCESS, error_code)
      << "GDF Operation did not complete successfully.";

  EXPECT_EQ(127, count);
}

TEST_F(ValidsTest, TenThousandRows) {
  const int num_rows = 10000;
  std::vector<int> data(num_rows);

  const int num_masks = std::ceil(num_rows / static_cast<float>(8));
  std::vector<gdf_valid_type> valid(num_masks, 0xFF);

  auto input_gdf_col = create_gdf_column(data, valid);

  int count{-1};
  gdf_error error_code =
      gdf_count_nonzero_mask(input_gdf_col->valid, num_rows, &count);

  ASSERT_EQ(GDF_SUCCESS, error_code)
      << "GDF Operation did not complete successfully.";

  EXPECT_EQ(10000, count);
}

TEST_F(ValidsTest, DISABLED_PerformanceTest) {
  const int num_rows = 100000000;
  std::vector<int> data(num_rows);

  const int num_masks = std::ceil(num_rows / 8);
  std::vector<gdf_valid_type> valid(num_masks, 0x55);

  auto input_gdf_col = create_gdf_column(data, valid);

  int count{-1};

  auto start = std::chrono::system_clock::now();
  cudaProfilerStart();
  for (int i = 0; i < 1000; ++i)
    gdf_error error_code =
        gdf_count_nonzero_mask(input_gdf_col->valid, num_rows, &count);
  cudaProfilerStop();
  auto end = std::chrono::system_clock::now();
  std::chrono::duration<double> elapsed_seconds = end - start;
  std::cout << "Elapsed time (ms): " << elapsed_seconds.count() * 1000
            << std::endl;
}

struct RowBitmaskTest : public GdfTest {};

template <typename Predicate>
struct validity_checker {
  validity_checker(Predicate _p, bit_mask::bit_mask_t* _bitmask)
      : p{_p}, bitmask{_bitmask} {}

  __device__ inline bool operator()(gdf_size_type index) {
    return p(index) == bit_mask::is_valid(bitmask, index);
  }

  bit_mask::bit_mask_t* bitmask;
  Predicate p;
};

struct all_valid {
  __host__ __device__ inline bool operator()(gdf_size_type index) {
    return true;
  }
};

struct all_null {
  __host__ __device__ inline bool operator()(gdf_size_type index) {
    return false;
  }
};

TEST_F(RowBitmaskTest, NoBitmasks) {
  gdf_size_type const size{1000};
  cudf::test::column_wrapper<int> col0{size};
  cudf::test::column_wrapper<float> col1{size};
  cudf::test::column_wrapper<double> col2{size};
  std::vector<gdf_column*> gdf_cols{col0.get(), col1.get(), col2.get()};
  cudf::table table{gdf_cols.data(),
                    static_cast<gdf_size_type>(gdf_cols.size())};

  rmm::device_vector<bit_mask::bit_mask_t> row_mask = cudf::row_bitmask(table);

  bool result = thrust::all_of(
      thrust::device, thrust::make_counting_iterator(0),
      thrust::make_counting_iterator(size),
      validity_checker<all_valid>(all_valid{}, row_mask.data().get()));

  EXPECT_TRUE(result);
}

TEST_F(RowBitmaskTest, BitmasksAllNull) {
  gdf_size_type const size{1000};
  cudf::test::column_wrapper<int> col0{size, true};
  cudf::test::column_wrapper<float> col1{size, true};
  cudf::test::column_wrapper<double> col2{size, true};
  std::vector<gdf_column*> gdf_cols{col0.get(), col1.get(), col2.get()};
  cudf::table table{gdf_cols.data(),
                    static_cast<gdf_size_type>(gdf_cols.size())};

  rmm::device_vector<bit_mask::bit_mask_t> row_mask = cudf::row_bitmask(table);

  bool result = thrust::all_of(
      thrust::device, thrust::make_counting_iterator(0),
      thrust::make_counting_iterator(size),
      validity_checker<all_null>(all_null{}, row_mask.data().get()));

  EXPECT_TRUE(result);
}

TEST_F(RowBitmaskTest, BitmasksAllValid) {
  gdf_size_type const size{1000};
  cudf::test::column_wrapper<int> col0{size, [](gdf_size_type i) { return i; },
                                       all_valid{}};
  cudf::test::column_wrapper<int> col1{size, [](gdf_size_type i) { return i; },
                                       all_valid{}};
  cudf::test::column_wrapper<int> col2{size, [](gdf_size_type i) { return i; },
                                       all_valid{}};
  std::vector<gdf_column*> gdf_cols{col0.get(), col1.get(), col2.get()};
  cudf::table table{gdf_cols.data(),
                    static_cast<gdf_size_type>(gdf_cols.size())};

  rmm::device_vector<bit_mask::bit_mask_t> row_mask = cudf::row_bitmask(table);

  bool result = thrust::all_of(
      thrust::device, thrust::make_counting_iterator(0),
      thrust::make_counting_iterator(size),
      validity_checker<all_valid>(all_valid{}, row_mask.data().get()));

  EXPECT_TRUE(result);
}

TEST_F(RowBitmaskTest, MixedBitmaskNoBitmaskAllValid) {
  gdf_size_type const size{1000};
  cudf::test::column_wrapper<int> col0{size};
  cudf::test::column_wrapper<int> col1{size, [](gdf_size_type i) { return i; },
                                       all_valid{}};
  cudf::test::column_wrapper<int> col2{size};

  std::vector<gdf_column*> gdf_cols{col0.get(), col1.get(), col2.get()};
  cudf::table table{gdf_cols.data(),
                    static_cast<gdf_size_type>(gdf_cols.size())};

  rmm::device_vector<bit_mask::bit_mask_t> row_mask = cudf::row_bitmask(table);

  bool result = thrust::all_of(
      thrust::device, thrust::make_counting_iterator(0),
      thrust::make_counting_iterator(size),
      validity_checker<all_valid>(all_valid{}, row_mask.data().get()));

  EXPECT_TRUE(result);
}

struct odds_are_null {
  __device__ inline bool operator()(gdf_size_type i) { return i % 2; }
};

TEST_F(RowBitmaskTest, MixedBitmaskNoBitmaskOddsNull) {
  gdf_size_type const size{1000};
  cudf::test::column_wrapper<int> col0{size};
  cudf::test::column_wrapper<int> col1{size, [](gdf_size_type i) { return i; },
                                       all_valid{}};
  cudf::test::column_wrapper<int> col2{size, [](gdf_size_type i) { return i; },
                                       [](gdf_size_type i) { return i % 2; }};

  std::vector<gdf_column*> gdf_cols{col0.get(), col1.get(), col2.get()};
  cudf::table table{gdf_cols.data(),
                    static_cast<gdf_size_type>(gdf_cols.size())};

  rmm::device_vector<bit_mask::bit_mask_t> row_mask = cudf::row_bitmask(table);

  bool result = thrust::all_of(
      thrust::device, thrust::make_counting_iterator(0),
      thrust::make_counting_iterator(size),
      validity_checker<odds_are_null>(odds_are_null{}, row_mask.data().get()));

  EXPECT_TRUE(result);
}
