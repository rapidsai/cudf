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

namespace {

template <typename T>
void expect_symmetry(cudf::column_view const& col)
{
  std::vector<T> host(col.size());
  cudaMemcpy(host.data(), col.data<T>(),
    col.size() * sizeof(T), cudaMemcpyDefault);
  CUDF_EXPECTS(std::equal(host.begin(), host.begin() + (col.size() / 2),
    host.rbegin()), "Expected hash to have symmetrical equality");
}

}

class HashTest : public cudf::test::BaseFixture {};

TEST_F(HashTest, MultiValueMurmur)
{
  using cudf::test::fixed_width_column_wrapper;
  using cudf::test::strings_column_wrapper;
  using cudf::experimental::bool8;

  std::vector<cudf::column_view> columns;

  auto const strings_col = strings_column_wrapper(
    {"",
    "The quick brown fox",
    "jumps over the lazy dog.",
    "All work and no play makes Jack a dull boy",
    "All work and no play makes Jack a dull boy",
    "jumps over the lazy dog.",
    "The quick brown fox",
    ""});
  columns.push_back(strings_col);

  auto const ints_col = fixed_width_column_wrapper<int32_t>(
    {0, 123, -456789, 123456789, 123456789, -456789, 123, 0});
  columns.push_back(ints_col);

  auto const bools_col = fixed_width_column_wrapper<bool8>(
    {0, 1, 10, 255, 255, 1, 1, 0});
  columns.push_back(bools_col);

  auto const secs_col = fixed_width_column_wrapper<cudf::timestamp_s>(
    {0, -123, 456, 123456, 123456, 456, -123, 0});
  columns.push_back(secs_col);

  // Expect output to have symmetrical equality
  auto const input = cudf::table_view(columns);
  auto const output = cudf::hash(input);
  expect_symmetry<int32_t>(output->view());
}

TEST_F(HashTest, MultiValueNullsMurmur)
{
  using cudf::test::fixed_width_column_wrapper;
  using cudf::test::strings_column_wrapper;
  using cudf::experimental::bool8;

  std::vector<cudf::column_view> columns;

  auto const strings_col = strings_column_wrapper(
    {"",
    "The quick brown fox",
    "jumps over the lazy dog.",
    "All work and no play makes Jack a dull boy",
    "All work and no play makes Jack a dull boy",
    "jumps over the lazy dog.",
    "The quick brown fox",
    ""},
    {0, 1, 1, 0, 0, 1, 1, 0});
  columns.push_back(strings_col);

  auto const ints_col = fixed_width_column_wrapper<int32_t>(
    {0, 123, -456789, 123456789, 123456789, -456789, 123, 0},
    {1, 0, 1, 0, 0, 1, 0, 1});
  columns.push_back(ints_col);

  auto const bools_col = fixed_width_column_wrapper<bool8>(
    {0, 1, 10, 255, 255, 1, 1, 0},
    {1, 1, 1, 0, 0, 1, 1, 1});
  columns.push_back(bools_col);

  auto const secs_col = fixed_width_column_wrapper<cudf::timestamp_s>(
    {0, -123, 456, 123456, 123456, 456, -123, 0},
    {1, 0, 1, 1, 1, 1, 0, 1});
  columns.push_back(secs_col);

  // Expect output to have symmetrical equality
  auto const input = cudf::table_view(columns);
  auto const output = cudf::hash(input);
  expect_symmetry<int32_t>(output->view());
}

/*template <typename T>
class HashTestTyped : public cudf::test::BaseFixture {};

TYPED_TEST_CASE(HashTestTyped, cudf::test::FixedWidthTypes);

TYPED_TEST(HashTestTyped, SingleValue)
{
}*/
