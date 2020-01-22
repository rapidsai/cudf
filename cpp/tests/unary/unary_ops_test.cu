/*
 * Copyright (c) 2019-2020, NVIDIA CORPORATION.
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

#include <cudf/cudf.h>
#include <cudf/column/column_factories.hpp>
#include <cudf/legacy/interop.hpp>
#include <cudf/types.hpp>
#include <cudf/unary.hpp>
#include <cudf/utilities/bit.hpp>
#include <cudf/utilities/type_dispatcher.hpp>
#include <cudf/wrappers/timestamps.hpp>
#include <initializer_list>
#include <tests/utilities/base_fixture.hpp>
#include <tests/utilities/column_utilities.hpp>
#include <tests/utilities/column_wrapper.hpp>
#include <tests/utilities/legacy/cudf_test_utils.cuh>
#include <tests/utilities/type_lists.hpp>
#include <vector>

template <typename T>
cudf::test::fixed_width_column_wrapper<T> create_fixed_columns(
    cudf::size_type start,
    cudf::size_type size,
    bool nullable) {
  auto iter = cudf::test::make_counting_transform_iterator(
      start, [](auto i) { return T(i); });

  if (not nullable) {
    return cudf::test::fixed_width_column_wrapper<T>(iter, iter + size);
  } else {
    auto valids = cudf::test::make_counting_transform_iterator(
        0, [](auto i) { return i % 2 == 0 ? true : false; });
    return cudf::test::fixed_width_column_wrapper<T>(iter, iter + size, valids);
  }
}

template <typename T>
cudf::test::fixed_width_column_wrapper<T>
create_expected_columns(cudf::size_type size, bool nullable, bool nulls_to_be) {
  if (not nullable) {
    auto iter = cudf::test::make_counting_transform_iterator(
        0, [nulls_to_be](auto i) { return not nulls_to_be; });
    return cudf::test::fixed_width_column_wrapper<T>(iter, iter + size);
  } else {
    auto iter =
        cudf::test::make_counting_transform_iterator(0, [nulls_to_be](auto i) {
          return i % 2 == 0 ? not nulls_to_be : nulls_to_be;
        });
    return cudf::test::fixed_width_column_wrapper<T>(iter, iter + size);
  }
}

template <typename T>
struct cudf_logical_test : public cudf::test::BaseFixture {};

TYPED_TEST_CASE(cudf_logical_test, cudf::test::NumericTypes);

TYPED_TEST(cudf_logical_test, LogicalNot) {
  cudf::size_type colSize = 1000;
  std::vector<TypeParam> h_input_v(colSize, false);
  std::vector<cudf::experimental::bool8> h_expect_v(colSize);

  std::transform(
      std::cbegin(h_input_v), std::cend(h_input_v), std::begin(h_expect_v),
      [](TypeParam e) -> cudf::bool8 { return static_cast<cudf::bool8>(!e); });

  cudf::test::fixed_width_column_wrapper<TypeParam> input(
      std::cbegin(h_input_v), std::cend(h_input_v));
  cudf::test::fixed_width_column_wrapper<cudf::experimental::bool8> expected(
      std::cbegin(h_expect_v), std::cend(h_expect_v));

  auto output = cudf::experimental::unary_operation(
      input, cudf::experimental::unary_op::NOT);

  cudf::test::expect_columns_equal(expected, output->view());
}

TYPED_TEST(cudf_logical_test, SimpleLogicalNot) {
  cudf::test::fixed_width_column_wrapper<TypeParam> input{
      {true, true, true, true}};
  cudf::test::fixed_width_column_wrapper<cudf::experimental::bool8> expected{
      {false, false, false, false}};
  auto output = cudf::experimental::unary_operation(
      input, cudf::experimental::unary_op::NOT);
  cudf::test::expect_columns_equal(expected, output->view());
}

TYPED_TEST(cudf_logical_test, SimpleLogicalNotWithNullMask) {
  cudf::test::fixed_width_column_wrapper<TypeParam> input{
      {true, true, true, true}, {1, 0, 1, 1}};
  cudf::test::fixed_width_column_wrapper<cudf::experimental::bool8> expected{
      {false, true, false, false}, {1, 0, 1, 1}};
  auto output = cudf::experimental::unary_operation(
      input, cudf::experimental::unary_op::NOT);
  cudf::test::expect_columns_equal(expected, output->view());
}

TYPED_TEST(cudf_logical_test, EmptyLogicalNot) {
  cudf::test::fixed_width_column_wrapper<TypeParam> input{};
  cudf::test::fixed_width_column_wrapper<cudf::experimental::bool8> expected{};
  auto output = cudf::experimental::unary_operation(
      input, cudf::experimental::unary_op::NOT);
  cudf::test::expect_columns_equal(expected, output->view());
}

template <typename T>
struct cudf_math_test : public cudf::test::BaseFixture {};

TYPED_TEST_CASE(cudf_math_test, cudf::test::NumericTypes);

TYPED_TEST(cudf_math_test, ABS) {
  using T = TypeParam;

  cudf::size_type colSize = 1000;
  std::vector<T> h_input_v(colSize);
  std::vector<T> h_expect_v(colSize);

  std::iota(std::begin(h_input_v), std::end(h_input_v), -1 * colSize);

  std::transform(std::cbegin(h_input_v), std::cend(h_input_v),
                 std::begin(h_expect_v), [](auto e) { return std::abs(e); });

  cudf::test::fixed_width_column_wrapper<T> input(std::cbegin(h_input_v),
                                                  std::cend(h_input_v));
  cudf::test::fixed_width_column_wrapper<T> expected(std::cbegin(h_expect_v),
                                                     std::cend(h_expect_v));

  auto output = cudf::experimental::unary_operation(
      input, cudf::experimental::unary_op::ABS);

  cudf::test::expect_columns_equal(expected, output->view());
}

TYPED_TEST(cudf_math_test, SQRT) {
  using T = TypeParam;

  cudf::size_type colSize = 1000;
  std::vector<T> h_input_v(colSize);
  std::vector<T> h_expect_v(colSize);

  std::generate(std::begin(h_input_v), std::end(h_input_v), [i = 0]() mutable {
    ++i;
    return i * i;
  });

  std::transform(std::cbegin(h_input_v), std::cend(h_input_v),
                 std::begin(h_expect_v),
                 [](auto e) { return std::sqrt(static_cast<float>(e)); });

  cudf::test::fixed_width_column_wrapper<T> input(std::cbegin(h_input_v),
                                                  std::cend(h_input_v));
  cudf::test::fixed_width_column_wrapper<T> expected(std::cbegin(h_expect_v),
                                                     std::cend(h_expect_v));

  auto output = cudf::experimental::unary_operation(
      input, cudf::experimental::unary_op::SQRT);

  cudf::test::expect_columns_equal(expected, output->view());
}

TYPED_TEST(cudf_math_test, SimpleABS) {
  cudf::test::fixed_width_column_wrapper<TypeParam> input{{-2, -1, 1, 2}};
  cudf::test::fixed_width_column_wrapper<TypeParam> expected{{2, 1, 1, 2}};
  auto output = cudf::experimental::unary_operation(
      input, cudf::experimental::unary_op::ABS);
  cudf::test::expect_columns_equal(expected, output->view());
}

TYPED_TEST(cudf_math_test, SimpleSQRT) {
  cudf::test::fixed_width_column_wrapper<TypeParam> input{{1, 4, 9, 16}};
  cudf::test::fixed_width_column_wrapper<TypeParam> expected{{1, 2, 3, 4}};
  auto output = cudf::experimental::unary_operation(
      input, cudf::experimental::unary_op::SQRT);
  cudf::test::expect_columns_equal(expected, output->view());
}

TYPED_TEST(cudf_math_test, SimpleSQRTWithNullMask) {
  cudf::test::fixed_width_column_wrapper<TypeParam> input{{1, 4, 9, 16},
                                                          {1, 1, 0, 1}};
  cudf::test::fixed_width_column_wrapper<TypeParam> expected{{1, 2, 9, 4},
                                                             {1, 1, 0, 1}};
  auto output = cudf::experimental::unary_operation(
      input, cudf::experimental::unary_op::SQRT);
  cudf::test::expect_columns_equal(expected, output->view());
}

TYPED_TEST(cudf_math_test, EmptyABS) {
  cudf::test::fixed_width_column_wrapper<TypeParam> input{};
  cudf::test::fixed_width_column_wrapper<TypeParam> expected{};
  auto output = cudf::experimental::unary_operation(
      input, cudf::experimental::unary_op::ABS);
  cudf::test::expect_columns_equal(expected, output->view());
}

TYPED_TEST(cudf_math_test, EmptySQRT) {
  cudf::test::fixed_width_column_wrapper<TypeParam> input{};
  cudf::test::fixed_width_column_wrapper<TypeParam> expected{};
  auto output = cudf::experimental::unary_operation(
      input, cudf::experimental::unary_op::SQRT);
  cudf::test::expect_columns_equal(expected, output->view());
}

template <typename T>
struct cudf_math_with_floating_point_test : public cudf::test::BaseFixture {};

using floating_point_type_list = ::testing::Types<float, double>;

TYPED_TEST_CASE(cudf_math_with_floating_point_test, floating_point_type_list);

TYPED_TEST(cudf_math_with_floating_point_test, SimpleSIN) {
  cudf::test::fixed_width_column_wrapper<TypeParam> input{{0.0}};
  cudf::test::fixed_width_column_wrapper<TypeParam> expected{{0.0}};
  auto output = cudf::experimental::unary_operation(
      input, cudf::experimental::unary_op::SIN);
  cudf::test::expect_columns_equal(expected, output->view());
}

TYPED_TEST(cudf_math_with_floating_point_test, SimpleCOS) {
  cudf::test::fixed_width_column_wrapper<TypeParam> input{{0.0}};
  cudf::test::fixed_width_column_wrapper<TypeParam> expected{{1.0}};
  auto output = cudf::experimental::unary_operation(
      input, cudf::experimental::unary_op::COS);
  cudf::test::expect_columns_equal(expected, output->view());
}

TYPED_TEST(cudf_math_with_floating_point_test, SimpleFLOOR) {
  cudf::test::fixed_width_column_wrapper<TypeParam> input{{1.1, 3.3, 5.5, 7.7}};
  cudf::test::fixed_width_column_wrapper<TypeParam> expected{
      {1.0, 3.0, 5.0, 7.0}};
  auto output = cudf::experimental::unary_operation(
      input, cudf::experimental::unary_op::FLOOR);
  cudf::test::expect_columns_equal(expected, output->view());
}

TYPED_TEST(cudf_math_with_floating_point_test, SimpleCEIL) {
  cudf::test::fixed_width_column_wrapper<TypeParam> input{{1.1, 3.3, 5.5, 7.7}};
  cudf::test::fixed_width_column_wrapper<TypeParam> expected{
      {2.0, 4.0, 6.0, 8.0}};
  auto output = cudf::experimental::unary_operation(
      input, cudf::experimental::unary_op::CEIL);
  cudf::test::expect_columns_equal(expected, output->view());
}

TYPED_TEST(cudf_math_with_floating_point_test, IntegralTypeFail) {
  cudf::test::fixed_width_column_wrapper<TypeParam> input{1.0};
  EXPECT_THROW(auto output = cudf::experimental::unary_operation(
                   input, cudf::experimental::unary_op::BIT_INVERT),
               cudf::logic_error);
}

template <typename T>
struct cudf_math_with_char_test : public cudf::test::BaseFixture {};

using just_char = ::testing::Types<char>;

TYPED_TEST_CASE(cudf_math_with_char_test, just_char);

TYPED_TEST(cudf_math_with_char_test, ArithmeticTypeFail) {
  cudf::test::fixed_width_column_wrapper<TypeParam> input{'c'};
  EXPECT_THROW(auto output = cudf::experimental::unary_operation(
                   input, cudf::experimental::unary_op::SQRT),
               cudf::logic_error);
}

TYPED_TEST(cudf_math_with_char_test, LogicalOpTypeFail) {
  cudf::test::fixed_width_column_wrapper<TypeParam> input{'h'};
  EXPECT_THROW(auto output = cudf::experimental::unary_operation(
                   input, cudf::experimental::unary_op::NOT),
               cudf::logic_error);
}

template <typename T>
struct IsNull : public cudf::test::BaseFixture {};

TYPED_TEST_CASE(IsNull, cudf::test::NumericTypes);

TYPED_TEST(IsNull, AllValid) {
  using T = TypeParam;

  cudf::size_type start = 0;
  cudf::size_type size = 10;
  cudf::test::fixed_width_column_wrapper<T> col =
      create_fixed_columns<T>(start, size, false);
  cudf::test::fixed_width_column_wrapper<cudf::experimental::bool8> expected =
      create_expected_columns<cudf::experimental::bool8>(size, false, true);

  std::unique_ptr<cudf::column> got = cudf::experimental::is_null(col);

  cudf::test::expect_columns_equal(expected, got->view());
}

TYPED_TEST(IsNull, WithInvalids) {
  using T = TypeParam;

  cudf::size_type start = 0;
  cudf::size_type size = 10;
  cudf::test::fixed_width_column_wrapper<T> col =
      create_fixed_columns<T>(start, size, true);
  cudf::test::fixed_width_column_wrapper<cudf::experimental::bool8> expected =
      create_expected_columns<cudf::experimental::bool8>(size, true, true);

  std::unique_ptr<cudf::column> got = cudf::experimental::is_null(col);

  cudf::test::expect_columns_equal(expected, got->view());
}

TYPED_TEST(IsNull, EmptyColumns) {
  using T = TypeParam;

  cudf::size_type start = 0;
  cudf::size_type size = 0;
  cudf::test::fixed_width_column_wrapper<T> col =
      create_fixed_columns<T>(start, size, true);
  cudf::test::fixed_width_column_wrapper<cudf::experimental::bool8> expected =
      create_expected_columns<cudf::experimental::bool8>(size, true, true);

  std::unique_ptr<cudf::column> got = cudf::experimental::is_null(col);

  cudf::test::expect_columns_equal(expected, got->view());
}

template <typename T>
struct IsNotNull : public cudf::test::BaseFixture {};

TYPED_TEST_CASE(IsNotNull, cudf::test::NumericTypes);

TYPED_TEST(IsNotNull, AllValid) {
  using T = TypeParam;

  cudf::size_type start = 0;
  cudf::size_type size = 10;
  cudf::test::fixed_width_column_wrapper<T> col =
      create_fixed_columns<T>(start, size, false);
  cudf::test::fixed_width_column_wrapper<cudf::experimental::bool8> expected =
      create_expected_columns<cudf::experimental::bool8>(size, false, false);

  std::unique_ptr<cudf::column> got = cudf::experimental::is_valid(col);

  cudf::test::expect_columns_equal(expected, got->view());
}

TYPED_TEST(IsNotNull, WithInvalids) {
  using T = TypeParam;

  cudf::size_type start = 0;
  cudf::size_type size = 10;
  cudf::test::fixed_width_column_wrapper<T> col =
      create_fixed_columns<T>(start, size, true);
  cudf::test::fixed_width_column_wrapper<cudf::experimental::bool8> expected =
      create_expected_columns<cudf::experimental::bool8>(size, true, false);

  std::unique_ptr<cudf::column> got = cudf::experimental::is_valid(col);

  cudf::test::expect_columns_equal(expected, got->view());
}

TYPED_TEST(IsNotNull, EmptyColumns) {
  using T = TypeParam;

  cudf::size_type start = 0;
  cudf::size_type size = 0;
  cudf::test::fixed_width_column_wrapper<T> col =
      create_fixed_columns<T>(start, size, true);
  cudf::test::fixed_width_column_wrapper<cudf::experimental::bool8> expected =
      create_expected_columns<cudf::experimental::bool8>(size, true, false);

  std::unique_ptr<cudf::column> got = cudf::experimental::is_valid(col);

  cudf::test::expect_columns_equal(expected, got->view());
}

static const auto test_timestamps_D = std::vector<int32_t>{
    -1528,  // 1965-10-26 GMT
    17716,  // 2018-07-04 GMT
    19382,  // 2023-01-25 GMT
};

static const auto test_timestamps_s = std::vector<int64_t>{
    -131968728,  // 1965-10-26 14:01:12 GMT
    1530705600,  // 2018-07-04 12:00:00 GMT
    1674631932,  // 2023-01-25 07:32:12 GMT
};

static const auto test_timestamps_ms = std::vector<int64_t>{
    -131968727238,  // 1965-10-26 14:01:12.762 GMT
    1530705600000,  // 2018-07-04 12:00:00.000 GMT
    1674631932929,  // 2023-01-25 07:32:12.929 GMT
};

static const auto test_timestamps_us = std::vector<int64_t>{
    -131968727238000,  // 1965-10-26 14:01:12.762000000 GMT
    1530705600000000,  // 2018-07-04 12:00:00.000000000 GMT
    1674631932929000,  // 2023-01-25 07:32:12.929000000 GMT
};

static const auto test_timestamps_ns = std::vector<int64_t>{
    -131968727238000000,  // 1965-10-26 14:01:12.762000000 GMT
    1530705600000000000,  // 2018-07-04 12:00:00.000000000 GMT
    1674631932929000000,  // 2023-01-25 07:32:12.929000000 GMT
};

template <typename T>
inline auto make_data_type() {
  return cudf::data_type{cudf::experimental::type_to_id<T>()};
}

template <typename T, typename R>
inline auto make_column(std::vector<R> data) {
  return cudf::test::fixed_width_column_wrapper<T>{data.begin(), data.end()};
}

template <typename T, typename R>
inline auto make_column(std::vector<R> data, std::vector<bool> mask) {
  return cudf::test::fixed_width_column_wrapper<T>{data.begin(), data.end(),
                                                   mask.begin()};
}

inline cudf::column make_exp_timestamp_column(cudf::type_id type_id) {
  switch (type_id) {
    case cudf::TIMESTAMP_DAYS:
      return cudf::column(
          cudf::data_type{type_id}, test_timestamps_D.size(),
          rmm::device_buffer{
              test_timestamps_D.data(),
              test_timestamps_D.size() * sizeof(test_timestamps_D.front())});
    case cudf::TIMESTAMP_SECONDS:
      return cudf::column(
          cudf::data_type{type_id}, test_timestamps_s.size(),
          rmm::device_buffer{
              test_timestamps_s.data(),
              test_timestamps_s.size() * sizeof(test_timestamps_s.front())});
    case cudf::TIMESTAMP_MILLISECONDS:
      return cudf::column(
          cudf::data_type{type_id}, test_timestamps_ms.size(),
          rmm::device_buffer{
              test_timestamps_ms.data(),
              test_timestamps_ms.size() * sizeof(test_timestamps_ms.front())});
    case cudf::TIMESTAMP_MICROSECONDS:
      return cudf::column(
          cudf::data_type{type_id}, test_timestamps_us.size(),
          rmm::device_buffer{
              test_timestamps_us.data(),
              test_timestamps_us.size() * sizeof(test_timestamps_us.front())});
    case cudf::TIMESTAMP_NANOSECONDS:
      return cudf::column(
          cudf::data_type{type_id}, test_timestamps_ns.size(),
          rmm::device_buffer{
              test_timestamps_ns.data(),
              test_timestamps_ns.size() * sizeof(test_timestamps_ns.front())});
    default:
      CUDF_FAIL("");
  }
};

template <typename T, typename R>
void validate_cast_result(cudf::column_view expected,
                          cudf::column_view actual) {
  using namespace cudf::test;
  // round-trip through the host because sizeof(T) may not equal sizeof(R)
  std::vector<T> h_data;
  std::vector<cudf::bitmask_type> null_mask;
  std::tie(h_data, null_mask) = to_host<T>(expected);
  if (null_mask.size() == 0) {
    expect_columns_equal(make_column<R, T>(h_data), actual);
  } else {
    std::vector<bool> h_null_mask{};
    for (cudf::size_type i = 0; i < expected.size(); ++i) {
      h_null_mask.push_back(cudf::bit_is_set(null_mask.data(), i));
    }
    expect_columns_equal(make_column<R, T>(h_data, h_null_mask), actual);
  }
}

struct CastTimestampsSimple : public cudf::test::BaseFixture {};

TEST_F(CastTimestampsSimple, IsIdempotent) {
  using namespace cudf::test;

  auto timestamps_D = make_column<cudf::timestamp_D>(test_timestamps_D);
  auto timestamps_s = make_column<cudf::timestamp_s>(test_timestamps_s);
  auto timestamps_ms = make_column<cudf::timestamp_ms>(test_timestamps_ms);
  auto timestamps_us = make_column<cudf::timestamp_us>(test_timestamps_us);
  auto timestamps_ns = make_column<cudf::timestamp_ns>(test_timestamps_ns);

  auto timestamps_D_rep = cudf::experimental::cast(
      timestamps_D, make_data_type<cudf::timestamp_D::rep>());
  auto timestamps_s_rep = cudf::experimental::cast(
      timestamps_s, make_data_type<cudf::timestamp_s::rep>());
  auto timestamps_ms_rep = cudf::experimental::cast(
      timestamps_ms, make_data_type<cudf::timestamp_ms::rep>());
  auto timestamps_us_rep = cudf::experimental::cast(
      timestamps_us, make_data_type<cudf::timestamp_us::rep>());
  auto timestamps_ns_rep = cudf::experimental::cast(
      timestamps_ns, make_data_type<cudf::timestamp_ns::rep>());

  auto timestamps_D_got = cudf::experimental::cast(
      *timestamps_D_rep, cudf::data_type{cudf::TIMESTAMP_DAYS});
  auto timestamps_s_got = cudf::experimental::cast(
      *timestamps_s_rep, cudf::data_type{cudf::TIMESTAMP_SECONDS});
  auto timestamps_ms_got = cudf::experimental::cast(
      *timestamps_ms_rep, cudf::data_type{cudf::TIMESTAMP_MILLISECONDS});
  auto timestamps_us_got = cudf::experimental::cast(
      *timestamps_us_rep, cudf::data_type{cudf::TIMESTAMP_MICROSECONDS});
  auto timestamps_ns_got = cudf::experimental::cast(
      *timestamps_ns_rep, cudf::data_type{cudf::TIMESTAMP_NANOSECONDS});

  validate_cast_result<cudf::timestamp_D, cudf::timestamp_D>(timestamps_D,
                                                             *timestamps_D_got);
  validate_cast_result<cudf::timestamp_s, cudf::timestamp_s>(timestamps_s,
                                                             *timestamps_s_got);
  validate_cast_result<cudf::timestamp_ms, cudf::timestamp_ms>(
      timestamps_ms, *timestamps_ms_got);
  validate_cast_result<cudf::timestamp_us, cudf::timestamp_us>(
      timestamps_us, *timestamps_us_got);
  validate_cast_result<cudf::timestamp_ns, cudf::timestamp_ns>(
      timestamps_ns, *timestamps_ns_got);
}

template <typename T>
struct CastTimestampsTyped : public cudf::test::BaseFixture {};

TYPED_TEST_CASE(CastTimestampsTyped, cudf::test::TimestampTypes);

// Return a list of timestamp type ids whose precision is greater than or equal
// to the input type id
std::vector<cudf::type_id> get_higher_precision_timestamp_type_ids(
    cudf::type_id search) {
  bool found = false;
  std::vector<cudf::type_id> gte_ids{};
  std::vector<cudf::type_id> timestamp_ids{
      cudf::TIMESTAMP_DAYS, cudf::TIMESTAMP_SECONDS,
      cudf::TIMESTAMP_MILLISECONDS, cudf::TIMESTAMP_MICROSECONDS,
      cudf::TIMESTAMP_NANOSECONDS};
  for (cudf::type_id type_id : timestamp_ids) {
    if (found || (found = (type_id == search))) {
      gte_ids.emplace_back(type_id);
    }
  }
  return gte_ids;
}

// Test that all timestamps whose precision is >= to the TypeParam
// down-casts appropriately to the lower-precision TypeParam
TYPED_TEST(CastTimestampsTyped, DownCastingFloorsValues) {
  using T = TypeParam;
  using namespace cudf::test;
  auto dtype_exp = make_data_type<T>();
  auto timestamps_exp = make_exp_timestamp_column(dtype_exp.id());
  // Construct a list of the timestamp type_ids whose precision is
  // greater than or equal to the precision of TypeParam's, e.g:
  // timestamp_ms -> {timestamp_ms, timestamp_us, timestamp_ns};
  // timestamp_us -> {timestamp_us, timestamp_ns};
  // etc.
  auto higher_precision_type_ids = get_higher_precision_timestamp_type_ids(
      cudf::experimental::type_to_id<T>());
  // For each higher-precision type, down-cast to TypeParam and validate
  // that the values were floored.
  for (cudf::type_id higher_precision_type_id : higher_precision_type_ids) {
    auto timestamps_src = make_exp_timestamp_column(higher_precision_type_id);
    auto timestamps_got = cudf::experimental::cast(timestamps_src, dtype_exp);
    expect_columns_equal(*timestamps_got, timestamps_exp);
  }
}

// Specific test to ensure down-casting to days happens correctly
TYPED_TEST(CastTimestampsTyped, DownCastingToDaysFloorsValues) {
  using T = TypeParam;
  using namespace cudf::test;

  auto dtype_src = make_data_type<T>();
  auto dtype_out = make_data_type<cudf::timestamp_D>();
  auto timestamps_src = make_exp_timestamp_column(dtype_src.id());
  auto timestamps_exp = make_column<cudf::timestamp_D>(test_timestamps_D);

  auto timestamps_got = cudf::experimental::cast(timestamps_src, dtype_out);

  validate_cast_result<cudf::timestamp_D, cudf::timestamp_D>(timestamps_exp,
                                                             *timestamps_got);
}

template <typename T>
struct CastToTimestamps : public cudf::test::BaseFixture {};

TYPED_TEST_CASE(CastToTimestamps, cudf::test::NumericTypes);

TYPED_TEST(CastToTimestamps, AllValid) {
  using T = TypeParam;
  using namespace cudf::test;

  auto timestamps_D = make_column<T>(test_timestamps_D);
  auto timestamps_s = make_column<T>(test_timestamps_s);
  auto timestamps_ms = make_column<T>(test_timestamps_ms);
  auto timestamps_us = make_column<T>(test_timestamps_us);
  auto timestamps_ns = make_column<T>(test_timestamps_ns);

  auto timestamps_D_got = cudf::experimental::cast(
      timestamps_D, cudf::data_type{cudf::TIMESTAMP_DAYS});
  auto timestamps_s_got = cudf::experimental::cast(
      timestamps_s, cudf::data_type{cudf::TIMESTAMP_SECONDS});
  auto timestamps_ms_got = cudf::experimental::cast(
      timestamps_ms, cudf::data_type{cudf::TIMESTAMP_MILLISECONDS});
  auto timestamps_us_got = cudf::experimental::cast(
      timestamps_us, cudf::data_type{cudf::TIMESTAMP_MICROSECONDS});
  auto timestamps_ns_got = cudf::experimental::cast(
      timestamps_ns, cudf::data_type{cudf::TIMESTAMP_NANOSECONDS});

  validate_cast_result<T, cudf::timestamp_D>(timestamps_D, *timestamps_D_got);
  validate_cast_result<T, cudf::timestamp_s>(timestamps_s, *timestamps_s_got);
  validate_cast_result<T, cudf::timestamp_ms>(timestamps_ms,
                                              *timestamps_ms_got);
  validate_cast_result<T, cudf::timestamp_us>(timestamps_us,
                                              *timestamps_us_got);
  validate_cast_result<T, cudf::timestamp_ns>(timestamps_ns,
                                              *timestamps_ns_got);
}

template <typename T>
struct CastFromTimestamps : public cudf::test::BaseFixture {};

TYPED_TEST_CASE(CastFromTimestamps, cudf::test::NumericTypes);

TYPED_TEST(CastFromTimestamps, AllValid) {
  using T = TypeParam;
  using namespace cudf::test;

  auto timestamps_D = make_column<cudf::timestamp_D>(test_timestamps_D);
  auto timestamps_s = make_column<cudf::timestamp_s>(test_timestamps_s);
  auto timestamps_ms = make_column<cudf::timestamp_ms>(test_timestamps_ms);
  auto timestamps_us = make_column<cudf::timestamp_us>(test_timestamps_us);
  auto timestamps_ns = make_column<cudf::timestamp_ns>(test_timestamps_ns);

  auto timestamps_D_exp = make_column<T>(test_timestamps_D);
  auto timestamps_s_exp = make_column<T>(test_timestamps_s);
  auto timestamps_ms_exp = make_column<T>(test_timestamps_ms);
  auto timestamps_us_exp = make_column<T>(test_timestamps_us);
  auto timestamps_ns_exp = make_column<T>(test_timestamps_ns);

  auto timestamps_D_got =
      cudf::experimental::cast(timestamps_D, make_data_type<T>());
  auto timestamps_s_got =
      cudf::experimental::cast(timestamps_s, make_data_type<T>());
  auto timestamps_ms_got =
      cudf::experimental::cast(timestamps_ms, make_data_type<T>());
  auto timestamps_us_got =
      cudf::experimental::cast(timestamps_us, make_data_type<T>());
  auto timestamps_ns_got =
      cudf::experimental::cast(timestamps_ns, make_data_type<T>());

  validate_cast_result<T, T>(timestamps_D_exp, *timestamps_D_got);
  validate_cast_result<T, T>(timestamps_s_exp, *timestamps_s_got);
  validate_cast_result<T, T>(timestamps_ms_exp, *timestamps_ms_got);
  validate_cast_result<T, T>(timestamps_us_exp, *timestamps_us_got);
  validate_cast_result<T, T>(timestamps_ns_exp, *timestamps_ns_got);
}

TYPED_TEST(CastFromTimestamps, WithNulls) {
  using T = TypeParam;
  using namespace cudf::test;

  auto timestamps_D =
      make_column<cudf::timestamp_D>(test_timestamps_D, {true, false, true});
  auto timestamps_s =
      make_column<cudf::timestamp_s>(test_timestamps_s, {true, false, true});
  auto timestamps_ms =
      make_column<cudf::timestamp_ms>(test_timestamps_ms, {true, false, true});
  auto timestamps_us =
      make_column<cudf::timestamp_us>(test_timestamps_us, {true, false, true});
  auto timestamps_ns =
      make_column<cudf::timestamp_ns>(test_timestamps_ns, {true, false, true});

  auto timestamps_D_exp =
      make_column<T>(test_timestamps_D, {true, false, true});
  auto timestamps_s_exp =
      make_column<T>(test_timestamps_s, {true, false, true});
  auto timestamps_ms_exp =
      make_column<T>(test_timestamps_ms, {true, false, true});
  auto timestamps_us_exp =
      make_column<T>(test_timestamps_us, {true, false, true});
  auto timestamps_ns_exp =
      make_column<T>(test_timestamps_ns, {true, false, true});

  auto timestamps_D_got =
      cudf::experimental::cast(timestamps_D, make_data_type<T>());
  auto timestamps_s_got =
      cudf::experimental::cast(timestamps_s, make_data_type<T>());
  auto timestamps_ms_got =
      cudf::experimental::cast(timestamps_ms, make_data_type<T>());
  auto timestamps_us_got =
      cudf::experimental::cast(timestamps_us, make_data_type<T>());
  auto timestamps_ns_got =
      cudf::experimental::cast(timestamps_ns, make_data_type<T>());

  validate_cast_result<T, T>(timestamps_D_exp, *timestamps_D_got);
  validate_cast_result<T, T>(timestamps_s_exp, *timestamps_s_got);
  validate_cast_result<T, T>(timestamps_ms_exp, *timestamps_ms_got);
  validate_cast_result<T, T>(timestamps_us_exp, *timestamps_us_got);
  validate_cast_result<T, T>(timestamps_ns_exp, *timestamps_ns_got);
}

template <typename T>
struct IsNAN : public cudf::test::BaseFixture {};

TYPED_TEST_CASE(IsNAN, cudf::test::FloatingPointTypes);

TYPED_TEST(IsNAN, AllValid) {
  using T = TypeParam;

  cudf::test::fixed_width_column_wrapper<T> col {{1, 2, NAN, 4, NAN, 6, 7}};
  cudf::test::fixed_width_column_wrapper<cudf::experimental::bool8> expected = {false, false, true, false, true, false, false};

  std::unique_ptr<cudf::column> got = cudf::experimental::is_nan(col);

  cudf::test::expect_columns_equal(expected, got->view());
}

TYPED_TEST(IsNAN, WithNull) {
  using T = TypeParam;

  // The last NAN is null
  cudf::test::fixed_width_column_wrapper<T> col {{1, 2, NAN, 4, NAN, 6, 7}, {1, 0, 1, 1, 0, 1, 1}};
  cudf::test::fixed_width_column_wrapper<cudf::experimental::bool8> expected = {false, false, true, false, false, false, false};

  std::unique_ptr<cudf::column> got = cudf::experimental::is_nan(col);

  cudf::test::expect_columns_equal(expected, got->view());
}

TYPED_TEST(IsNAN, EmptyColumn) {
  using T = TypeParam;

  cudf::test::fixed_width_column_wrapper<T> col {};
  cudf::test::fixed_width_column_wrapper<cudf::experimental::bool8> expected = {};

  std::unique_ptr<cudf::column> got = cudf::experimental::is_nan(col);

  cudf::test::expect_columns_equal(expected, got->view());
}

TYPED_TEST(IsNAN, NonFloatingColumn) {
  using T = TypeParam;

  cudf::test::fixed_width_column_wrapper<int32_t> col {{1, 2, 5, 3, 5, 6, 7}, {1, 0, 1, 1, 0, 1, 1}};

  EXPECT_THROW(std::unique_ptr<cudf::column> got = cudf::experimental::is_nan(col), cudf::logic_error);
}

template <typename T>
struct IsNotNAN : public cudf::test::BaseFixture {};

TYPED_TEST_CASE(IsNotNAN, cudf::test::FloatingPointTypes);

TYPED_TEST(IsNotNAN, AllValid) {
  using T = TypeParam;

  cudf::test::fixed_width_column_wrapper<T> col {{1, 2, NAN, 4, NAN, 6, 7}};
  cudf::test::fixed_width_column_wrapper<cudf::experimental::bool8> expected = {true, true, false, true, false, true, true};

  std::unique_ptr<cudf::column> got = cudf::experimental::is_not_nan(col);

  cudf::test::expect_columns_equal(expected, got->view());
}

TYPED_TEST(IsNotNAN, WithNull) {
  using T = TypeParam;

  // The last NAN is null
  cudf::test::fixed_width_column_wrapper<T> col {{1, 2, NAN, 4, NAN, 6, 7}, {1, 0, 1, 1, 0, 1, 1}};
  cudf::test::fixed_width_column_wrapper<cudf::experimental::bool8> expected = {true, true, false, true, true, true, true};

  std::unique_ptr<cudf::column> got = cudf::experimental::is_not_nan(col);

  cudf::test::expect_columns_equal(expected, got->view());
}

TYPED_TEST(IsNotNAN, EmptyColumn) {
  using T = TypeParam;

  cudf::test::fixed_width_column_wrapper<T> col {};
  cudf::test::fixed_width_column_wrapper<cudf::experimental::bool8> expected = {};

  std::unique_ptr<cudf::column> got = cudf::experimental::is_not_nan(col);

  cudf::test::expect_columns_equal(expected, got->view());
}

TYPED_TEST(IsNotNAN, NonFloatingColumn) {
  using T = TypeParam;

  cudf::test::fixed_width_column_wrapper<int64_t> col {{1, 2, 5, 3, 5, 6, 7}, {1, 0, 1, 1, 0, 1, 1}};

  EXPECT_THROW(std::unique_ptr<cudf::column> got = cudf::experimental::is_not_nan(col), cudf::logic_error);
}

