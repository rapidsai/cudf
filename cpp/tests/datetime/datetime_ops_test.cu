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
#include <cudf/column/column_factories.hpp>
#include <cudf/column/column_device_view.cuh>

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

template <typename Element>
void print_column(cudf::column_view col) {
  print_typed_column<Element>(
    col.data<Element>(),
    (gdf_valid_type*) col.null_mask(),
    col.size(),
    1);
}

// TYPED_TEST_CASE(DatetimeOpsTest, cudf::test::TimestampTypes);

TYPED_TEST_CASE(DatetimeOpsTest, cudf::test::Types<cudf::timestamp_D>);

TYPED_TEST(DatetimeOpsTest, TimestampDurationsMatchPrimitiveRepresentation) {

  using namespace cudf::test;
  using namespace simt::std::chrono;
  using Rep = typename TypeParam::rep;
  using Period = typename TypeParam::period;

  auto start = milliseconds(-2500000000000); // Sat, 11 Oct 1890 19:33:20 GMT
  auto stop_ = milliseconds( 2500000000000); // Mon, 22 Mar 2049 04:26:40 GMT
  auto test_timestamps = generate_timestamps<Rep, Period>(this->size(),
                                                          time_point_ms(start),
                                                          time_point_ms(stop_));

  auto timestamp_col = cudf::make_timestamp_column(this->type(), this->size(),
                                                   cudf::mask_state::UNALLOCATED,
                                                   this->stream(), this->mr());

  cudf::mutable_column_view timestamp_view = *timestamp_col;
  // auto timestamp_dev_view = cudf::column_device_view::create(timestamp_view);

  CUDA_TRY(cudaMemcpy(timestamp_view.data<Rep>(),
    thrust::raw_pointer_cast(test_timestamps.data()),
    test_timestamps.size() * sizeof(Rep), cudaMemcpyDefault));

  print_column<Rep>(timestamp_view);

  auto expected_years = fixed_width_column_wrapper<int16_t>{1890, 1906, 1922, 1938, 1954, 1970, 1985, 2001, 2017, 2033};
  auto actual_years = *cudf::datetime::extract_year(timestamp_view);
  print_column<int16_t>(expected_years);
  print_column<int16_t>(actual_years);
  expect_columns_equal(expected_years, actual_years);

  // auto expected_months = fixed_width_column_wrapper<int16_t>{9, 7, 5, 3, 1, 0, 10, 8, 6, 4};
  // auto actual_months = *cudf::datetime::extract_month(timestamp_view);
  // expect_columns_equal(expected_months, actual_months);

  // auto expected_days = fixed_width_column_wrapper<int16_t>{11, 16, 20, 24, 26, 1, 5, 9, 14, 18};
  // auto actual_days = *cudf::datetime::extract_day(timestamp_view);
  // expect_columns_equal(expected_days, actual_days);

  // auto expected_weekdays = fixed_width_column_wrapper<int16_t>{6, 4, 2, 0, 5, 4, 2, 0, 5, 3};
  // auto actual_weekdays = *cudf::datetime::extract_weekday(timestamp_view);
  // expect_columns_equal(expected_weekdays, actual_weekdays);

  // auto expected_hours = fixed_width_column_wrapper<int16_t>{19, 20, 21, 22, 23, 0, 0, 1, 2, 3};
  // auto actual_hours = *cudf::datetime::extract_hour(timestamp_view);
  // expect_columns_equal(expected_hours, actual_hours);

  // auto expected_minutes = fixed_width_column_wrapper<int16_t>{33, 26, 20, 13, 6, 0, 53, 46, 40, 33};
  // auto actual_minutes = *cudf::datetime::extract_minute(timestamp_view);
  // expect_columns_equal(expected_minutes, actual_minutes);

  // auto expected_seconds = fixed_width_column_wrapper<int16_t>{20, 40, 0, 20, 40, 0, 20, 40, 0, 20};
  // auto actual_seconds = *cudf::datetime::extract_second(timestamp_view);
  // expect_columns_equal(expected_seconds, actual_seconds);
}
