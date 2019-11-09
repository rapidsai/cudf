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
#include <cudf/hashing.hpp>
#include <tests/utilities/base_fixture.hpp>
#include <tests/utilities/type_lists.hpp>
#include <tests/utilities/column_utilities.hpp>
#include <tests/utilities/column_wrapper.hpp>

// for expect_columns_not_equal
#include <cudf/table/table_device_view.cuh>
#include <cudf/table/row_operators.cuh>

namespace {

// TODO maybe refactor expect_columns_equal to return bool instead of assert
void expect_columns_not_equal(cudf::column_view lhs, cudf::column_view rhs) {
  // NOTE columns properties SHOULD be equal
  cudf::test::expect_column_properties_equal(lhs, rhs);

  auto d_lhs = cudf::table_device_view::create(cudf::table_view{{lhs}});
  auto d_rhs = cudf::table_device_view::create(cudf::table_view{{rhs}});

  EXPECT_FALSE(
      thrust::equal(thrust::device, thrust::make_counting_iterator(0),
                    thrust::make_counting_iterator(lhs.size()),
                    thrust::make_counting_iterator(0),
                    cudf::experimental::row_equality_comparator<true>{*d_lhs, *d_rhs}));

  // TODO is this necessary?
  CUDA_TRY(cudaDeviceSynchronize());
}

}

class HashTest : public cudf::test::BaseFixture {};

TEST_F(HashTest, MultiValue)
{
  using cudf::test::fixed_width_column_wrapper;
  using cudf::test::strings_column_wrapper;
  using cudf::test::expect_columns_equal;
  using cudf::experimental::bool8;

  auto const strings_col = strings_column_wrapper(
    {"",
    "The quick brown fox",
    "jumps over the lazy dog.",
    "All work and no play makes Jack a dull boy",
    "!\"#$%&\'()*+,-./0123456789:;<=>?@[\\]^_`{|}~"});

  using limits = std::numeric_limits<int32_t>;
  auto const ints_col = fixed_width_column_wrapper<int32_t>(
    {0, 100, -100, limits::min(), limits::max()});

  // Different truthy values should be equal
  auto const bools_col1 = fixed_width_column_wrapper<bool8>({0, 1, 1, 1, 0});
  auto const bools_col2 = fixed_width_column_wrapper<bool8>({0, 1, 2, 255, 0});

  using ts = cudf::timestamp_s;
  auto const secs_col = fixed_width_column_wrapper<ts>(
    {ts::duration::zero(), 100, -100, ts::duration::min(), ts::duration::max()});

  auto const input1 = cudf::table_view(
    {strings_col, ints_col, bools_col1, secs_col});
  auto const input2 = cudf::table_view(
    {strings_col, ints_col, bools_col2, secs_col});

  auto const output1 = cudf::hash(input1);
  auto const output2 = cudf::hash(input2);

  expect_columns_equal(output1->view(), output2->view());
}

TEST_F(HashTest, MultiValueNulls)
{
  using cudf::test::fixed_width_column_wrapper;
  using cudf::test::strings_column_wrapper;
  using cudf::test::expect_columns_equal;
  using cudf::experimental::bool8;

  // Nulls with different values should be equal
  auto const strings_col1 = strings_column_wrapper(
    {"",
    "The quick brown fox",
    "jumps over the lazy dog.",
    "All work and no play makes Jack a dull boy",
    "!\"#$%&\'()*+,-./0123456789:;<=>?@[\\]^_`{|}~"},
    {0, 1, 1, 0, 1});
  auto const strings_col2 = strings_column_wrapper(
    {"different but null",
    "The quick brown fox",
    "jumps over the lazy dog.",
    "I am Jack's complete lack of null value",
    "!\"#$%&\'()*+,-./0123456789:;<=>?@[\\]^_`{|}~"},
    {0, 1, 1, 0, 1});

  // Nulls with different values should be equal
  using limits = std::numeric_limits<int32_t>;
  auto const ints_col1 = fixed_width_column_wrapper<int32_t>(
    {0, 100, -100, limits::min(), limits::max()}, {1, 0, 0, 1, 1});
  auto const ints_col2 = fixed_width_column_wrapper<int32_t>(
    {0, -200, 200, limits::min(), limits::max()}, {1, 0, 0, 1, 1});

  // Nulls with different values should be equal
  // Different truthy values should be equal
  auto const bools_col1 = fixed_width_column_wrapper<bool8>(
    {0, 1, 0, 1, 1}, {1, 1, 0, 0, 1});
  auto const bools_col2 = fixed_width_column_wrapper<bool8>(
    {0, 2, 1, 0, 255}, {1, 1, 0, 0, 1});

  // Nulls with different values should be equal
  using ts = cudf::timestamp_s;
  auto const secs_col1 = fixed_width_column_wrapper<ts>(
    {ts::duration::zero(), 100, -100, ts::duration::min(), ts::duration::max()},
    {1, 0, 0, 1, 1});
  auto const secs_col2 = fixed_width_column_wrapper<ts>(
    {ts::duration::zero(), -200, 200, ts::duration::min(), ts::duration::max()},
    {1, 0, 0, 1, 1});

  auto const input1 = cudf::table_view(
    {strings_col1, ints_col1, bools_col1, secs_col1});
  auto const input2 = cudf::table_view(
    {strings_col2, ints_col2, bools_col2, secs_col2});

  auto const output1 = cudf::hash(input1);
  auto const output2 = cudf::hash(input2);

  expect_columns_equal(output1->view(), output2->view());
}

TEST_F(HashTest, StringInequality)
{
  using cudf::test::strings_column_wrapper;

  auto const strings_col1 = strings_column_wrapper(
    {"",
    "The quick brown fox",
    "jumps over the lazy dog.",
    "All work and no play makes Jack a dull boy",
    "!\"#$%&\'()*+,-./0123456789:;<=>?@[\\]^_`{|}~"});
  auto const strings_col2 = strings_column_wrapper(
    {" ",
    "ThE quIck broWn foX",
    "jumpS over the laZy dog.",
    "All work anD no play maKes Jack a duLl boy",
    "!\"#$%&\'()*+,-./0123456789:;<=>?@[\\]^_`{|}~"});

  auto const input1 = cudf::table_view({strings_col1});
  auto const input2 = cudf::table_view({strings_col2});

  auto const output1 = cudf::hash(input1);
  auto const output2 = cudf::hash(input2);

  expect_columns_not_equal(output1->view(), output2->view());
}

/*template <typename T>
class HashTestTyped : public cudf::test::BaseFixture {};

TYPED_TEST_CASE(HashTestTyped, cudf::test::FixedWidthTypes);

TYPED_TEST(HashTestTyped, SingleValue)
{
}*/
