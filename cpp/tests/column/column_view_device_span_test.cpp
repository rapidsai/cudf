/*
 * Copyright (c) 2022-2024, NVIDIA CORPORATION.
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
#include <cudf_test/column_utilities.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/type_lists.hpp>

#include <cudf/column/column_view.hpp>
#include <cudf/null_mask.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/span.hpp>
#include <cudf/utilities/traits.hpp>

#include <thrust/iterator/counting_iterator.h>

#include <memory>

template <typename T, CUDF_ENABLE_IF(cudf::is_numeric<T>() or cudf::is_chrono<T>())>
std::unique_ptr<cudf::column> example_column()
{
  auto begin = thrust::make_counting_iterator(1);
  auto end   = thrust::make_counting_iterator(16);
  return cudf::test::fixed_width_column_wrapper<T>(begin, end).release();
}

template <typename T>
struct ColumnViewDeviceSpanTests : public cudf::test::BaseFixture {};

using DeviceSpanTypes = cudf::test::FixedWidthTypesWithoutFixedPoint;
TYPED_TEST_SUITE(ColumnViewDeviceSpanTests, DeviceSpanTypes);

TYPED_TEST(ColumnViewDeviceSpanTests, conversion_round_trip)
{
  auto col      = example_column<TypeParam>();
  auto col_view = cudf::column_view{*col};

  // Test implicit conversion, round trip
  cudf::device_span<TypeParam const> device_span_from_col_view = col_view;
  cudf::column_view col_view_from_device_span                  = device_span_from_col_view;
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(col_view, col_view_from_device_span);
}

struct ColumnViewDeviceSpanErrorTests : public cudf::test::BaseFixture {};

TEST_F(ColumnViewDeviceSpanErrorTests, type_mismatch)
{
  auto col      = example_column<int32_t>();
  auto col_view = cudf::column_view{*col};
  EXPECT_THROW((void)cudf::device_span<float const>{col_view}, cudf::logic_error);
}

TEST_F(ColumnViewDeviceSpanErrorTests, nullable_column)
{
  auto col = example_column<int32_t>();
  col->set_null_mask(cudf::create_null_mask(col->size(), cudf::mask_state::ALL_NULL), col->size());
  auto col_view = cudf::column_view{*col};
  EXPECT_THROW((void)cudf::device_span<int32_t const>{col_view}, cudf::logic_error);
}
