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

#include <tests/utilities/column_wrapper.hpp>
#include <tests/utilities/cudf_gtest.hpp>
#include <tests/utilities/base_fixture.hpp>
#include <tests/utilities/column_utilities.hpp>
#include <tests/utilities/type_lists.hpp>

#include <cudf/copying.hpp>
#include <cudf/column/column.hpp>
#include <cudf/column/column_view.hpp>

#include <thrust/iterator/counting_iterator.h>

auto all_valid = [](cudf::size_type row) { return true; };
auto even_valid = [](cudf::size_type row) { return (row % 2 == 0); };

template <typename T>
class CopyRangeTypedTestFixture : public cudf::test::BaseFixture {
public:
  static constexpr auto column_size = cudf::size_type{1000};

  void test(cudf::mutable_column_view& destination,
            cudf::column_view const& source,
            cudf::column_view const& expected,
            cudf::size_type out_begin, cudf::size_type out_end,
            cudf::size_type in_begin) {
    static_assert(cudf::is_fixed_width<T>());

    // test the out-of-place version first

    const auto immutable_view = cudf::column_view{destination};
    auto ret = cudf::experimental::copy_range(immutable_view, source,
                                              out_begin, out_end, in_begin);
    cudf::test::expect_columns_equal(*ret, expected);

    // test the in-place version second

    EXPECT_NO_THROW(cudf::experimental::copy_range(destination, source,
                    out_begin, out_end, in_begin));
    cudf::test::expect_columns_equal(destination, expected);
  }
};

TYPED_TEST_CASE(CopyRangeTypedTestFixture, cudf::test::FixedWidthTypes);

TYPED_TEST(CopyRangeTypedTestFixture, CopyWithNulls)
{
  using T = TypeParam;

  auto size = cudf::size_type{CopyRangeTypedTestFixture<T>::column_size};
  auto out_begin = cudf::size_type{30};
  auto out_end = cudf::size_type{size - 20};
  auto in_begin = cudf::size_type{9};
  auto row_diff = in_begin - out_begin;

  auto destination = cudf::test::fixed_width_column_wrapper<T>(
    thrust::make_counting_iterator(0),
    thrust::make_counting_iterator(0) + size,
    cudf::test::make_counting_transform_iterator(0, all_valid));

  auto source_elements =
    cudf::test::make_counting_transform_iterator(
      0, [](auto i) { return i * 2; });
  auto source = cudf::test::fixed_width_column_wrapper<T>(
    source_elements,
    source_elements + size,
    cudf::test::make_counting_transform_iterator(
      0, even_valid));

  auto expected_elements =
    cudf::test::make_counting_transform_iterator(
      0,
      [out_begin, out_end, row_diff](auto i) {
        return ((i >= out_begin) && (i < out_end)) ?
          (i + row_diff) * 2 : i;
     });
  auto expected = cudf::test::fixed_width_column_wrapper<T>(
    expected_elements,
    expected_elements + size,
    cudf::test::make_counting_transform_iterator(
      0,
      [out_begin, out_end, row_diff](auto i) {
        return ((i >= out_begin) && (i < out_end)) ?
          even_valid(i + row_diff) : all_valid(i);
      }));

  auto destination_view = cudf::mutable_column_view{destination};
  this->test(destination_view, source, expected, out_begin, out_end, in_begin);
}

TYPED_TEST(CopyRangeTypedTestFixture, CopyNoNulls)
{
  using T = TypeParam;

  auto size = cudf::size_type{CopyRangeTypedTestFixture<T>::column_size};
  auto out_begin = cudf::size_type{30};
  auto out_end = cudf::size_type{size - 20};
  auto in_begin = cudf::size_type{9};
  auto row_diff = in_begin - out_begin;

  auto destination = cudf::test::fixed_width_column_wrapper<T>(
    thrust::make_counting_iterator(0),
    thrust::make_counting_iterator(0) + size);

  auto source_elements =
    cudf::test::make_counting_transform_iterator(
      0, [](auto i) { return i * 2; });
  auto source = cudf::test::fixed_width_column_wrapper<T>(
    source_elements,
    source_elements + size);

  auto expected_elements =
    cudf::test::make_counting_transform_iterator(
      0,
      [out_begin, out_end, row_diff](auto i) {
        return ((i >= out_begin) && (i < out_end)) ?
          (i + row_diff) * 2 : i;
     });
  auto expected = cudf::test::fixed_width_column_wrapper<T>(
    expected_elements,
    expected_elements + size);

  auto destination_view = cudf::mutable_column_view{destination};
  this->test(destination_view, source, expected, out_begin, out_end, in_begin);
}

class CopyRangeErrorTestFixture : public cudf::test::BaseFixture {};

TEST_F(CopyRangeErrorTestFixture, InvalidInplaceCall)
{
  auto size = cudf::size_type{100};

  auto destination = cudf::test::fixed_width_column_wrapper<int32_t>(
    thrust::make_counting_iterator(0),
    thrust::make_counting_iterator(0) + size);

  auto source = cudf::test::fixed_width_column_wrapper<int32_t>(
    thrust::make_counting_iterator(0),
    thrust::make_counting_iterator(0) + size,
    cudf::test::make_counting_transform_iterator(0, even_valid));

  auto destination_view = cudf::mutable_column_view{destination};
  // source has null values but destination is not nullable.
  EXPECT_THROW(cudf::experimental::copy_range(destination_view, source,
                                              0, size, 0),
               cudf::logic_error);

  auto strings =
    std::vector<std::string>{"", "this", "is", "a", "column", "of", "strings"};
  auto destination_string =
    cudf::test::strings_column_wrapper(strings.begin(), strings.end());
  auto source_string =
    cudf::test::strings_column_wrapper(strings.begin(), strings.end());

  auto destination_view_string = cudf::mutable_column_view{destination_string};
  EXPECT_THROW(cudf::experimental::copy_range(
                 destination_view_string, source_string, 0, size, 0),
               cudf::logic_error);
}

TEST_F(CopyRangeErrorTestFixture, InvalidRange)
{
  auto size = cudf::size_type{100};

  auto destination = cudf::test::fixed_width_column_wrapper<int32_t>(
    thrust::make_counting_iterator(0),
    thrust::make_counting_iterator(0) + size);

  auto source = cudf::test::fixed_width_column_wrapper<int32_t>(
    thrust::make_counting_iterator(0),
    thrust::make_counting_iterator(0) + size);

  auto destination_view = cudf::mutable_column_view{destination};

  // empty_range == no-op, this is valid
  EXPECT_NO_THROW(cudf::experimental::copy_range(
                 destination_view, source, 0, 0, 0));
  EXPECT_NO_THROW(auto ret = cudf::experimental::copy_range(
                 destination, source, 0, 0, 0));

  // out_begin is negative
  EXPECT_THROW(cudf::experimental::copy_range(
                 destination_view, source, -1, size, 0),
               cudf::logic_error);
  EXPECT_THROW(auto ret = cudf::experimental::copy_range(
                 destination, source, -1, size, 0),
               cudf::logic_error);

  // out_begin > out_end
  EXPECT_THROW(cudf::experimental::copy_range(
                 destination_view, source, 10, 5, 0),
               cudf::logic_error);
  EXPECT_THROW(auto ret = cudf::experimental::copy_range(
                 destination, source, 10, 5, 0),
               cudf::logic_error);

  // out_begin >= destination.size()
  EXPECT_THROW(cudf::experimental::copy_range(
                 destination_view, source, 100, 100, 0),
               cudf::logic_error);
  EXPECT_THROW(auto ret = cudf::experimental::copy_range(
                 destination, source, 100, 100, 0),
               cudf::logic_error);

  // out_end > destination.size()
  EXPECT_THROW(cudf::experimental::copy_range(
                 destination_view, source, 99, 101, 0),
               cudf::logic_error);
  EXPECT_THROW(auto ret = cudf::experimental::copy_range(
                 destination, source, 99, 101, 0),
               cudf::logic_error);

  // in_begin < 0
  EXPECT_THROW(cudf::experimental::copy_range(
                 destination_view, source, 50, 100, -5),
               cudf::logic_error);
  EXPECT_THROW(auto ret = cudf::experimental::copy_range(
                 destination, source, 50, 100, -5),
               cudf::logic_error);

  // in_begin >= source.size()
  EXPECT_THROW(cudf::experimental::copy_range(
                 destination_view, source, 50, 100, 100),
               cudf::logic_error);
  EXPECT_THROW(auto ret = cudf::experimental::copy_range(
                 destination, source, 50, 100, 100),
               cudf::logic_error);

  // in_begin + out_end - out_begin > source.size()
  EXPECT_THROW(cudf::experimental::copy_range(
                 destination_view, source, 50, 100, 80),
               cudf::logic_error);
  EXPECT_THROW(auto ret = cudf::experimental::copy_range(
                 destination, source, 50, 100, 80),
               cudf::logic_error);
}

TEST_F(CopyRangeErrorTestFixture, DTypeMismatch)
{
  auto size = cudf::size_type{100};

  auto destination = cudf::test::fixed_width_column_wrapper<int32_t>(
    thrust::make_counting_iterator(0),
    thrust::make_counting_iterator(0) + size);

  auto source = cudf::test::fixed_width_column_wrapper<float>(
    thrust::make_counting_iterator(0),
    thrust::make_counting_iterator(0) + size);

  auto destination_view = cudf::mutable_column_view{destination};

  EXPECT_THROW(cudf::experimental::copy_range(
                 destination_view, source, 0, 100, 0),
               cudf::logic_error);
  EXPECT_THROW(auto ret = cudf::experimental::copy_range(
                 destination, source, 0, 100, 0),
               cudf::logic_error);
}

TEST_F(CopyRangeErrorTestFixture, StringCategoryNotSupported)
{
  auto strings =
    std::vector<std::string>{"", "this", "is", "a", "column", "of", "strings"};
  auto destination_string =
    cudf::test::strings_column_wrapper(strings.begin(), strings.end());
  auto source_string =
    cudf::test::strings_column_wrapper(strings.begin(), strings.end());

  EXPECT_THROW(auto ret = cudf::experimental::copy_range(
                 destination_string, source_string, 0, 1, 0),
               cudf::logic_error);
}
