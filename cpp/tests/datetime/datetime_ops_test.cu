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

#include <cudf/datetime.hpp>
#include <cudf/utilities/chrono.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/column/column_factories.hpp>

#include <tests/utilities/type_lists.hpp>
#include <tests/utilities/base_fixture.hpp>
#include <tests/utilities/column_wrapper.hpp>
#include <tests/utilities/column_utilities.hpp>
#include <tests/utilities/timestamp_utilities.cuh>

#include <tests/utilities/legacy/cudf_test_utils.cuh>

#include <gmock/gmock.h>

template <typename T>
struct DatetimeOpsTest : public cudf::test::BaseFixture {
  cudaStream_t stream() { return cudaStream_t(0); }
  cudf::size_type size() { return cudf::size_type(10); }
  cudf::data_type type() { return cudf::data_type{cudf::experimental::type_to_id<T>()}; }
};

TYPED_TEST_CASE(DatetimeOpsTest, cudf::test::TimestampTypes);

TYPED_TEST(DatetimeOpsTest, TestExtractingDatetimeComponents) {

  using namespace cudf::test;
  using namespace simt::std::chrono;

  auto test_timestamps_D = fixed_width_column_wrapper<cudf::timestamp_D>{
    -1528, // 1965-10-26
    17716, // 2018-07-04
    19382, // 2023-01-25
  };

  auto test_timestamps_s = fixed_width_column_wrapper<cudf::timestamp_s>{
    -131968728, // 1965-10-26 14:01:12
    1530705600, // 2018-07-04 12:00:00
    1674631932, // 2023-01-25 07:32:12
  };

  auto test_timestamps_ms = fixed_width_column_wrapper<cudf::timestamp_ms>{
    -131968727238, // 1965-10-26 14:01:12.762
    1530705600000, // 2018-07-04 12:00:00.000
    1674631932929, // 2023-01-25 07:32:12.929
  };

  expect_columns_equal(*cudf::datetime::extract_year(test_timestamps_D), fixed_width_column_wrapper<int16_t>{1965, 2018, 2023});
  expect_columns_equal(*cudf::datetime::extract_year(test_timestamps_s), fixed_width_column_wrapper<int16_t>{1965, 2018, 2023});
  expect_columns_equal(*cudf::datetime::extract_year(test_timestamps_ms), fixed_width_column_wrapper<int16_t>{1965, 2018, 2023});

  expect_columns_equal(*cudf::datetime::extract_month(test_timestamps_D), fixed_width_column_wrapper<int16_t>{10, 7, 1});
  expect_columns_equal(*cudf::datetime::extract_month(test_timestamps_s), fixed_width_column_wrapper<int16_t>{10, 7, 1});
  expect_columns_equal(*cudf::datetime::extract_month(test_timestamps_ms), fixed_width_column_wrapper<int16_t>{10, 7, 1});

  expect_columns_equal(*cudf::datetime::extract_day(test_timestamps_D), fixed_width_column_wrapper<int16_t>{26, 4, 25});
  expect_columns_equal(*cudf::datetime::extract_day(test_timestamps_s), fixed_width_column_wrapper<int16_t>{26, 4, 25});
  expect_columns_equal(*cudf::datetime::extract_day(test_timestamps_ms), fixed_width_column_wrapper<int16_t>{26, 4, 25});

  expect_columns_equal(*cudf::datetime::extract_weekday(test_timestamps_D), fixed_width_column_wrapper<int16_t>{2, 3, 3});
  expect_columns_equal(*cudf::datetime::extract_weekday(test_timestamps_s), fixed_width_column_wrapper<int16_t>{2, 3, 3});
  expect_columns_equal(*cudf::datetime::extract_weekday(test_timestamps_ms), fixed_width_column_wrapper<int16_t>{2, 3, 3});

  expect_columns_equal(*cudf::datetime::extract_hour(test_timestamps_D), fixed_width_column_wrapper<int16_t>{0, 0, 0});
  expect_columns_equal(*cudf::datetime::extract_hour(test_timestamps_s), fixed_width_column_wrapper<int16_t>{14, 12, 7});
  expect_columns_equal(*cudf::datetime::extract_hour(test_timestamps_ms), fixed_width_column_wrapper<int16_t>{14, 12, 7});

  expect_columns_equal(*cudf::datetime::extract_minute(test_timestamps_D), fixed_width_column_wrapper<int16_t>{0, 0, 0});
  expect_columns_equal(*cudf::datetime::extract_minute(test_timestamps_s), fixed_width_column_wrapper<int16_t>{1, 0, 32});
  expect_columns_equal(*cudf::datetime::extract_minute(test_timestamps_ms), fixed_width_column_wrapper<int16_t>{1, 0, 32});

  expect_columns_equal(*cudf::datetime::extract_second(test_timestamps_D), fixed_width_column_wrapper<int16_t>{0, 0, 0});
  expect_columns_equal(*cudf::datetime::extract_second(test_timestamps_s), fixed_width_column_wrapper<int16_t>{12, 0, 12});
  expect_columns_equal(*cudf::datetime::extract_second(test_timestamps_ms), fixed_width_column_wrapper<int16_t>{12, 0, 12});

}

TYPED_TEST(DatetimeOpsTest, TestExtractingGeneratedDatetimeComponents) {

  using T = TypeParam;
  using namespace cudf::test;
  using namespace simt::std::chrono;

  auto start = milliseconds(-2500000000000); // Sat, 11 Oct 1890 19:33:20 GMT
  auto stop_ = milliseconds( 2500000000000); // Mon, 22 Mar 2049 04:26:40 GMT
  auto timestamp_col = generate_timestamps<T>(this->size(),
                                              time_point_ms(start),
                                              time_point_ms(stop_));

  auto expected_years = fixed_width_column_wrapper<int16_t>{1890, 1906, 1922, 1938, 1954, 1970, 1985, 2001, 2017, 2033};
  auto expected_months = fixed_width_column_wrapper<int16_t>{10, 8, 6, 4, 2, 1, 11, 9, 7, 5};
  auto expected_days = fixed_width_column_wrapper<int16_t>{11, 16, 20, 24, 26, 1, 5, 9, 14, 18};
  auto expected_weekdays = fixed_width_column_wrapper<int16_t>{6, 4, 2, 7, 5, 4, 2, 7, 5, 3};
  auto expected_hours = fixed_width_column_wrapper<int16_t>{19, 20, 21, 22, 23, 0, 0, 1, 2, 3};
  auto expected_minutes = fixed_width_column_wrapper<int16_t>{33, 26, 20, 13, 6, 0, 53, 46, 40, 33};
  auto expected_seconds = fixed_width_column_wrapper<int16_t>{20, 40, 0, 20, 40, 0, 20, 40, 0, 20};

  // Special cases for timestamp_D: zero out the hh/mm/ss cols and +1 the expected weekdays
  if (std::is_same<TypeParam, cudf::timestamp_D>::value) {
    expected_days = fixed_width_column_wrapper<int16_t>{12, 17, 21, 25, 27, 1, 5, 9, 14, 18};
    expected_weekdays = fixed_width_column_wrapper<int16_t>{7, 5, 3, 1, 6, 4, 2, 7, 5, 3};
    expected_hours = fixed_width_column_wrapper<int16_t>{0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    expected_minutes = fixed_width_column_wrapper<int16_t>{0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    expected_seconds = fixed_width_column_wrapper<int16_t>{0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  }

  expect_columns_equal(*cudf::datetime::extract_year(timestamp_col), expected_years);
  expect_columns_equal(*cudf::datetime::extract_month(timestamp_col), expected_months);
  expect_columns_equal(*cudf::datetime::extract_day(timestamp_col), expected_days);
  expect_columns_equal(*cudf::datetime::extract_weekday(timestamp_col), expected_weekdays);
  expect_columns_equal(*cudf::datetime::extract_hour(timestamp_col), expected_hours);
  expect_columns_equal(*cudf::datetime::extract_minute(timestamp_col), expected_minutes);
  expect_columns_equal(*cudf::datetime::extract_second(timestamp_col), expected_seconds);
}
