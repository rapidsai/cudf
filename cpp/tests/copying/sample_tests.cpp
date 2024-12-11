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

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/table_utilities.hpp>

#include <cudf/copying.hpp>
#include <cudf/detail/iterator.cuh>
#include <cudf/sorting.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>

struct SampleTest : public cudf::test::BaseFixture {};

TEST_F(SampleTest, FailCaseRowMultipleSampling)
{
  cudf::test::fixed_width_column_wrapper<int16_t> col1{1, 2, 3, 4, 5};
  // size of input table is smaller than number of samples
  // and sampling same row multiple times is disallowed,
  // this combination can't work.
  cudf::size_type const n_samples = 10;
  cudf::table_view input({col1});

  EXPECT_THROW(cudf::sample(input, n_samples, cudf::sample_with_replacement::FALSE, 0),
               cudf::logic_error);
}

TEST_F(SampleTest, RowMultipleSamplingDisallowed)
{
  cudf::size_type const n_samples = 1024;
  auto data = cudf::detail::make_counting_transform_iterator(0, [](auto i) { return i; });
  cudf::test::fixed_width_column_wrapper<int16_t> col1(data, data + n_samples);

  cudf::table_view input({col1});

  for (int i = 0; i < 10; i++) {
    auto out_table  = cudf::sample(input, n_samples, cudf::sample_with_replacement::FALSE, i);
    auto sorted_out = cudf::sort(out_table->view());

    CUDF_TEST_EXPECT_TABLES_EQUAL(input, sorted_out->view());
  }
}

TEST_F(SampleTest, TestReproducibilityWithSeed)
{
  cudf::size_type const n_samples = 1024;
  auto data = cudf::detail::make_counting_transform_iterator(0, [](auto i) { return i; });
  cudf::test::fixed_width_column_wrapper<int16_t> col1(data, data + n_samples);

  cudf::table_view input({col1});

  auto expected_1 = cudf::sample(input, n_samples, cudf::sample_with_replacement::FALSE, 1);
  for (int i = 0; i < 2; i++) {
    auto out = cudf::sample(input, n_samples, cudf::sample_with_replacement::FALSE, 1);

    CUDF_TEST_EXPECT_TABLES_EQUAL(expected_1->view(), out->view());
  }

  auto expected_2 = cudf::sample(input, n_samples, cudf::sample_with_replacement::TRUE, 1);
  for (int i = 0; i < 2; i++) {
    auto out = cudf::sample(input, n_samples, cudf::sample_with_replacement::TRUE, 1);

    CUDF_TEST_EXPECT_TABLES_EQUAL(expected_2->view(), out->view());
  }
}

struct SampleBasicTest : public SampleTest,
                         public ::testing::WithParamInterface<
                           std::tuple<cudf::size_type, cudf::sample_with_replacement>> {};

TEST_P(SampleBasicTest, CombinationOfParameters)
{
  cudf::size_type const table_size   = 1024;
  auto const [n_samples, multi_smpl] = GetParam();

  auto data = cudf::detail::make_counting_transform_iterator(0, [](auto i) { return i; });
  cudf::test::fixed_width_column_wrapper<int16_t> col1(data, data + table_size);
  cudf::test::fixed_width_column_wrapper<float> col2(data, data + table_size);

  cudf::table_view input({col1, col2});
  auto out_table = cudf::sample(input, n_samples, multi_smpl, 0);

  EXPECT_EQ(out_table->num_rows(), n_samples);
}

INSTANTIATE_TEST_CASE_P(
  SampleTest,
  SampleBasicTest,
  ::testing::Values(std::make_tuple(0, cudf::sample_with_replacement::TRUE),
                    std::make_tuple(0, cudf::sample_with_replacement::FALSE),
                    std::make_tuple(512, cudf::sample_with_replacement::TRUE),
                    std::make_tuple(512, cudf::sample_with_replacement::FALSE),
                    std::make_tuple(1024, cudf::sample_with_replacement::TRUE),
                    std::make_tuple(1024, cudf::sample_with_replacement::FALSE),
                    std::make_tuple(2048, cudf::sample_with_replacement::TRUE)));
